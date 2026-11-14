//===- DoConcurrentConversion.cpp -- map `DO CONCURRENT` to OpenMP loops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/DirectivesCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "flang/Optimizer/OpenMP/Utils.h"
#include "flang/Support/OpenMP-utils.h"
#include "flang/Utils/OpenMP.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

namespace flangomp {
#define GEN_PASS_DEF_DOCONCURRENTCONVERSIONPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "do-concurrent-conversion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {
namespace looputils {
/// Stores info needed about the induction/iteration variable for each `do
/// concurrent` in a loop nest.
struct InductionVariableInfo {
  InductionVariableInfo(fir::DoConcurrentLoopOp loop,
                        mlir::Value inductionVar) {
    populateInfo(loop, inductionVar);
  }
  /// The operation allocating memory for iteration variable.
  mlir::Operation *iterVarMemDef;
  /// the operation(s) updating the iteration variable with the current
  /// iteration number.
  llvm::SmallVector<mlir::Operation *, 2> indVarUpdateOps;

private:
  /// For the \p doLoop parameter, find the following:
  ///
  /// 1. The operation that declares its iteration variable or allocates memory
  /// for it. For example, give the following loop:
  /// ```
  ///   ...
  ///   %i:2 = hlfir.declare %0 {uniq_name = "_QFEi"} : ...
  ///   ...
  ///   fir.do_concurrent.loop (%ind_var) = (%lb) to (%ub) step (%s) {
  ///     %ind_var_conv = fir.convert %ind_var : (index) -> i32
  ///     fir.store %ind_var_conv to %i#1 : !fir.ref<i32>
  ///     ...
  ///   }
  /// ```
  ///
  /// This function sets the `iterVarMemDef` member to the `hlfir.declare` op
  /// for `%i`.
  ///
  /// 2. The operation(s) that update the loop's iteration variable from its
  /// induction variable. For the above example, the `indVarUpdateOps` is
  /// populated with the first 2 ops in the loop's body.
  ///
  /// Note: The current implementation is dependent on how flang emits loop
  /// bodies; which is sufficient for the current simple test/use cases. If this
  /// proves to be insufficient, this should be made more generic.
  void populateInfo(fir::DoConcurrentLoopOp loop, mlir::Value inductionVar) {
    mlir::Value result = nullptr;

    // Checks if a StoreOp is updating the memref of the loop's iteration
    // variable.
    auto isStoringIV = [&](fir::StoreOp storeOp) {
      // Direct store into the IV memref.
      if (storeOp.getValue() == inductionVar) {
        indVarUpdateOps.push_back(storeOp);
        return true;
      }

      // Indirect store into the IV memref.
      if (auto convertOp = mlir::dyn_cast<fir::ConvertOp>(
              storeOp.getValue().getDefiningOp())) {
        if (convertOp.getOperand() == inductionVar) {
          indVarUpdateOps.push_back(convertOp);
          indVarUpdateOps.push_back(storeOp);
          return true;
        }
      }

      return false;
    };

    for (mlir::Operation &op : loop) {
      if (auto storeOp = mlir::dyn_cast<fir::StoreOp>(op))
        if (isStoringIV(storeOp)) {
          result = storeOp.getMemref();
          break;
        }
    }

    assert(result != nullptr && result.getDefiningOp() != nullptr);
    iterVarMemDef = result.getDefiningOp();
  }
};

using InductionVariableInfos = llvm::SmallVector<InductionVariableInfo>;

/// Collect the list of values used inside the loop but defined outside of it.
void collectLoopLiveIns(fir::DoConcurrentLoopOp loop,
                        llvm::SmallVectorImpl<mlir::Value> &liveIns) {
  llvm::SmallDenseSet<mlir::Value> seenValues;
  llvm::SmallPtrSet<mlir::Operation *, 8> seenDeclares;

  for (auto [lb, ub, st] : llvm::zip_equal(
           loop.getLowerBound(), loop.getUpperBound(), loop.getStep())) {
    liveIns.push_back(lb);
    liveIns.push_back(ub);
    liveIns.push_back(st);
  }

  mlir::visitUsedValuesDefinedAbove(
      loop.getRegion(), [&](mlir::OpOperand *operand) {
        mlir::Value value = operand->get();
        if (!seenValues.insert(value).second)
          return;

        // Both results of an `hlfir.declare` are views of the same variable,
        // which is mapped once. Results of other multi-result ops (e.g.
        // `fir.box_dims`) are independent values and each must be mapped.
        if (auto declareOp = value.getDefiningOp<hlfir::DeclareOp>())
          if (!seenDeclares.insert(declareOp).second)
            return;

        liveIns.push_back(value);
      });

  for (mlir::Value local : loop.getLocalVars())
    liveIns.push_back(local);

  for (mlir::Value reduce : loop.getReduceVars())
    liveIns.push_back(reduce);
}

/// Collects values that are local to a loop: "loop-local values". A loop-local
/// value is one that is used exclusively inside the loop but allocated outside
/// of it. This usually corresponds to temporary values that are used inside the
/// loop body for initialzing other variables for example.
///
/// See `flang/test/Transforms/DoConcurrent/locally_destroyed_temp.f90` for an
/// example of why we need this.
///
/// \param [in] doLoop - the loop within which the function searches for values
/// used exclusively inside.
///
/// \param [out] locals - the list of loop-local values detected for \p doLoop.
void collectLoopLocalValues(fir::DoConcurrentLoopOp loop,
                            llvm::SetVector<mlir::Value> &locals) {
  loop.walk([&](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands()) {
      if (locals.contains(operand))
        continue;

      bool isLocal = true;

      if (!mlir::isa_and_present<fir::AllocaOp>(operand.getDefiningOp()))
        continue;

      // Values defined inside the loop are not interesting since they do not
      // need to be localized.
      if (loop->isAncestor(operand.getDefiningOp()))
        continue;

      for (auto *user : operand.getUsers()) {
        if (!loop->isAncestor(user)) {
          isLocal = false;
          break;
        }
      }

      if (isLocal)
        locals.insert(operand);
    }
  });
}

/// For a "loop-local" value \p local within a loop's scope, localizes that
/// value within the scope of the parallel region the loop maps to. Towards that
/// end, this function moves the allocation of \p local within \p allocRegion.
///
/// \param local - the value used exclusively within a loop's scope (see
/// collectLoopLocalValues).
///
/// \param allocRegion - the parallel region where \p local's allocation will be
/// privatized.
///
/// \param rewriter - builder used for updating \p allocRegion.
static void localizeLoopLocalValue(mlir::Value local, mlir::Region &allocRegion,
                                   mlir::ConversionPatternRewriter &rewriter) {
  rewriter.moveOpBefore(local.getDefiningOp(), &allocRegion.front().front());
}

/// Holds information needed to inline a LOCAL_INIT copy_region into the
/// omp.loop_nest body for per-iteration initialization.
///
/// Fortran LOCAL_INIT semantics require that at the start of each iteration,
/// the private variable is re-initialized from the host (original) variable.
/// OpenMP firstprivate (omp.private copy_region) only initializes once per
/// thread, which is wrong. Instead, we:
///   (A) Create the omp.private privatizer the same way as LOCAL (no
///       copy_region on the privatizer — just init + dealloc).
///   (B) Inline the copy_region ops at the top of the omp.loop_nest body so
///       re-initialization happens every iteration.
///
/// This struct carries the info collected during (A) in genPrivatizers() that
/// is consumed during (B) in matchAndRewrite() after block arg remapping.
///
/// Example of what the generated IR looks like (host path):
///
///   omp.private {type = private} @tmp.omp : f32  // no copy_region
///
///   omp.parallel {
///     omp.wsloop private(@tmp.omp %orig -> %priv : !fir.ref<f32>) {
///       omp.loop_nest (%iv) = (%lb) to (%ub) step (%st) {
///         // --- inlined copy_region (per-iteration) ---
///         %val = fir.load %orig : !fir.ref<f32>
///         fir.store %val to %priv : !fir.ref<f32>
///         // --- end inlined copy ---
///         ... loop body ...
///         omp.yield
///       }
///     }
///     omp.terminator
///   }
struct LocalInitCopyInfo {
  /// The fir.local op whose copy_region will be inlined. The copy_region
  /// contains the operations that initialize the private variable from the
  /// original (e.g., load from original, store to private).
  fir::LocalitySpecifierOp localizer;

  /// The host (original) variable that the copy reads from. When the loop is
  /// not moved into a new target region (`remapsLiveIns` is false), this is
  /// the raw SSA value from the enclosing scope. Otherwise it is the
  /// device-mapped value obtained via mapper.lookup().
  mlir::Value originalVar;

  /// Index into the privateClauseOps.privateVars list. After genEntryBlock()
  /// creates block args for the wsloop (host) or parallel (device) op, this
  /// index resolves to the actual block arg representing private storage.
  unsigned privateVarIdx;
};

/// Inlines the operations from a fir.local copy_region into an existing block,
/// at the beginning of that block. This is used for LOCAL_INIT to achieve
/// per-iteration initialization within the omp.loop_nest body.
///
/// The copy_region has two block arguments:
///   - arg0: the original (mold) variable to read from
///   - arg1: the private variable to write to
///
/// These are mapped to \p originalVar and \p privateVar respectively via an
/// IRMapping. Each operation in the copy_region (except the fir.yield
/// terminator) is cloned into the target block.
///
/// This is modeled on the cloneLocalizerRegion lambda in
/// SimplifyFIROperations.cpp, which does the same thing for the non-OpenMP
/// path. We cannot use cloneFIRRegionToOMP or rewriter.cloneRegionBefore
/// because those would create a new block in the target region, but
/// omp.loop_nest needs its body in a single block.
///
/// For a simple scalar f32 LOCAL_INIT, the copy_region typically contains:
///   ^bb0(%orig: !fir.ref<f32>, %priv: !fir.ref<f32>):
///     %val = fir.load %orig : !fir.ref<f32>
///     fir.store %val to %priv : !fir.ref<f32>
///     fir.yield(%priv : !fir.ref<f32>)
///
/// After inlining into the loop body (with mapping orig->originalVar,
/// priv->privateVar), the loop body starts with:
///     %val = fir.load %originalVar : !fir.ref<f32>
///     fir.store %val to %privateVar : !fir.ref<f32>
///     ... rest of loop body ...
static mlir::LogicalResult
inlineCopyRegionToLoopBody(mlir::ConversionPatternRewriter &rewriter,
                           mlir::Region &copyRegion, mlir::Value originalVar,
                           mlir::Value privateVar, mlir::Block &loopBody) {
  // At this point in the pipeline (before CFG conversion), control-flow
  // constructs like `if` are still represented by their encapsulating
  // structured FIR ops (e.g. fir.if), so regions are expected to have a
  // single block containing straight-line code. Multi-block copy regions
  // could theoretically arise if CFG conversion ran earlier or if a complex
  // type's copy involves lowered control flow, but this is not expected at
  // this pipeline stage. If it does happen, bail out with a diagnostic
  // rather than silently miscompiling.
  //
  // TODO: Support multi-block copy regions if the need arises. This would
  // require inlining the full CFG (multiple blocks with branches) into the
  // loop body, not just cloning ops from a single block.
  if (!copyRegion.hasOneBlock())
    return mlir::emitError(copyRegion.getLoc())
           << "LOCAL_INIT copy region with multiple blocks is not yet "
              "supported in OpenMP conversion. Expected single-block "
              "straight-line code at this pipeline stage.";

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&loopBody);

  // Map the copy_region's block arguments to the concrete SSA values:
  //   copy_region arg0 (mold/original) -> originalVar
  //   copy_region arg1 (private)       -> privateVar
  mlir::IRMapping mapping;
  mapping.map(copyRegion.getArgument(0), originalVar);
  mapping.map(copyRegion.getArgument(1), privateVar);

  // Clone each operation except the terminator (fir.yield). The fir.yield's
  // result is not needed — the copy is purely side-effecting (load + store).
  for (mlir::Operation &op : copyRegion.front().without_terminator())
    rewriter.clone(op, mapping);

  return mlir::success();
}

} // namespace looputils

/// Returns true if \p value is an integer constant, possibly wrapped in
/// `fir.convert` ops.
static bool isIntConstant(mlir::Value value) {
  while (auto convertOp = value.getDefiningOp<fir::ConvertOp>())
    value = convertOp.getValue();
  return fir::getIntIfConstant(value).has_value();
}

/// Describes *what* the pattern emits for one particular `do concurrent` loop.
///
/// The `-fdo-concurrent-to-openmp=host|device` option only states the user's
/// request (stored in `DoConcurrentConversion::mapToDevice`). Which OpenMP
/// construct nest is actually generated for a given loop is a separate,
/// per-loop decision, made once at the start of `matchAndRewrite` by
/// `DoConcurrentConversion::computeEmissionContext` and then threaded through
/// all the `gen*` helpers.
///
/// Keeping the request and the decision apart lets later changes emit
/// different nests for loops in different surroundings (for example a loop
/// that is already inside an `omp.target` region) without re-testing the raw
/// option in every helper. Today the mapping is one-to-one:
///
///   -fdo-concurrent-to-openmp=host   -> EmissionContext::Host
///   -fdo-concurrent-to-openmp=device -> EmissionContext::NewDeviceTarget
///
/// A context value is never inspected directly by the code that builds the
/// OpenMP ops. Instead, four predicate functions defined right after this enum
/// each map a context to one boolean property of the nest to generate:
///
///   createsTargetRegion    - create a new omp.target (with map_entries for
///                            the live-ins and host_eval loop bounds)?
///   remapsLiveIns          - must host SSA values be replaced by their
///                            device-side counterparts (via the IRMapping)
///                            before being used as clause operands?
///   distributesAcrossTeams - emit omp.teams + omp.distribute, making the nest
///                            a composite `distribute parallel do`?
///   privatizesOnParallel   - attach `private` clauses to omp.parallel
///                            instead of omp.wsloop?
///
/// The code in `matchAndRewrite` and the `gen*` helpers only calls these
/// predicates (e.g. `if (createsTargetRegion(ctx)) ...`). These four
/// properties used to be implied together by the single `mapToDevice` flag;
/// keeping them separate allows a context in which, for instance, teams are
/// distributed but no new target region is created. Adding a new context
/// therefore means adding an enumerator, returning it from
/// `computeEmissionContext`, and defining the value of each of the four
/// predicates for it. The op-building code only needs changes where it still
/// ties two properties together (for example, omp.teams is currently created
/// together with the new omp.target).
enum class EmissionContext {
  /// Parallelize on the host CPU. Generates (simplified):
  ///
  ///   omp.parallel {                       // omp.combined
  ///     omp.wsloop private(...) reduction(...) {
  ///       omp.loop_nest (...) { ... }
  ///     }
  ///   }
  ///
  /// Live-ins are used directly: `omp.parallel` is not IsolatedFromAbove, so
  /// host values remain visible inside the region (i.e. they are shared).
  Host,

  /// Offload to a device with a *new* target region created by the pass.
  /// Generates (simplified):
  ///
  ///   omp.target host_eval(lb, ub, step) map_entries(live-ins...) {
  ///     omp.teams reduction(...) {         // target and teams: omp.combined
  ///       omp.parallel private(...) {      // omp.composite
  ///         omp.distribute {               // omp.composite
  ///           omp.wsloop reduction(...) {  // omp.composite
  ///             omp.loop_nest (...) { ... }
  ///   } } } } }
  ///
  /// `omp.target` is IsolatedFromAbove, so every live-in is mapped through an
  /// `omp.map.info` op and host values are remapped (via an `IRMapping`) to the
  /// corresponding target block arguments.
  NewDeviceTarget,

  /// Offload requested, but the loop is already directly inside a target
  /// region written by the user (e.g. `!$omp target` around the loop), and
  /// apart from the loop, the region only contains code that can be executed
  /// once per team without changing the program's behavior: computations
  /// without side effects, memory reads, and variable declarations (see
  /// `canEmitTeamsInTarget`). Generates the same nest as `NewDeviceTarget`
  /// but *inside the existing target*, without creating a new one:
  ///
  ///   omp.target kernel_type(generic) ... { // user's op, now omp.combined
  ///     ...declarations (and code computing their shapes), constants...
  ///     omp.teams reduction(...) {          // omp.combined
  ///       omp.parallel private(...) {       // omp.composite
  ///         ...bound computations, re-evaluated here
  ///            (see cloneTargetBodyValuesIntoParallel)...
  ///         omp.distribute {                // omp.composite
  ///           omp.wsloop reduction(...) {   // omp.composite
  ///             omp.loop_nest (...) { ... }
  ///   } } } } }
  ///
  /// Why not just create a nested `omp.target`? OpenMP does not allow a target
  /// construct to be encountered inside a target region (unspecified in 4.5 /
  /// 5.0, prohibited since 5.1 except for reverse offload), and in practice the
  /// nested region silently does not run the loop.
  ///
  /// The user's target op gets the `omp.combined` marker (a target whose only
  /// construct is a teams region must have it, as lowering does for
  /// `!$omp target` + nested `!$omp teams`), but keeps its kernel type.
  ///
  /// Why keep the user's target `generic` instead of upgrading it to `spmd`?
  /// An `spmd` kernel needs host-evaluated loop bounds, but here the bounds
  /// are computed inside the target region. LLVM's OpenMPOpt pass converts
  /// such generic kernels to "Generic-SPMD" at -O1 and above, and the runtime
  /// launches many teams because of the `omp.teams` op (number of compute
  /// units x a per-mode factor) even without a host trip count. Measured,
  /// this is as fast as the fully combined form.
  ///
  /// No live-in mapping is needed: the new ops are created inside the user's
  /// target region, where every value the loop uses is already a device value.
  TeamsInUserTarget,

  /// Offload requested and the loop is inside a target region written by the
  /// user, but a teams nest is not possible there: the loop is not directly in
  /// the target's body (e.g. it is inside an `if` or a sequential loop in the
  /// target), or the target also contains other code that must not run once
  /// per team (stores, calls, allocations, another loop). OpenMP requires a
  /// `teams` construct nested in a `target` to be the only thing in it.
  ///
  /// Falls back to the same nest as `Host`, placed inside the target region:
  ///
  ///   omp.target ... {                        // user's op, left untouched
  ///     ...
  ///     omp.parallel {                        // omp.combined
  ///       omp.wsloop private(...) reduction(...) {
  ///         omp.loop_nest (...) { ... }
  ///   } } }
  ///
  /// This is always legal and correct, but it runs on a single team.
  ParallelInUserTarget,
};

/// Returns true if the pass creates a new `omp.target` region for the loop.
/// This also implies mapping live-ins with `omp.map.info` ops and evaluating
/// the loop bounds on the host (`host_eval`).
static bool createsTargetRegion(EmissionContext ctx) {
  return ctx == EmissionContext::NewDeviceTarget;
}

/// Returns true if host-side SSA values used by the loop must be looked up in
/// the host-to-device `IRMapping` before being used as clause operands (e.g.
/// privatized or reduced variables). This is the case exactly when the loop
/// is moved into a new, IsolatedFromAbove `omp.target` region.
static bool remapsLiveIns(EmissionContext ctx) {
  return createsTargetRegion(ctx);
}

/// Returns true if the generated nest distributes iterations across teams,
/// i.e. it contains `omp.teams` and `omp.distribute`. In that case
/// `omp.parallel`, `omp.distribute` and `omp.wsloop` together form the
/// composite `distribute parallel do` construct and are marked
/// `omp.composite`; otherwise `omp.parallel` + `omp.wsloop` form the combined
/// `parallel do` construct and `omp.parallel` is marked `omp.combined`.
static bool distributesAcrossTeams(EmissionContext ctx) {
  return ctx == EmissionContext::NewDeviceTarget ||
         ctx == EmissionContext::TeamsInUserTarget;
}

/// Returns true if `private` clauses (for LOCAL / LOCAL_INIT variables) are
/// attached to `omp.parallel`, false if they are attached to `omp.wsloop`.
///
/// In a `distribute parallel do` composite nest, `omp.distribute` sits between
/// `omp.parallel` and `omp.wsloop`, and the privatized storage must be
/// allocated per thread by the parallel region that encloses both of them.
/// In a plain `parallel do` nest the worksharing loop owns the privates.
static bool privatizesOnParallel(EmissionContext ctx) {
  return distributesAcrossTeams(ctx);
}

/// Returns true if \p op can be executed once per team instead of once
/// without changing the program's results: \p op (including any ops nested
/// in its regions) either has no memory effects, only reads memory, or only
/// touches flang's `fir::DebuggingResource` (which is how `hlfir.declare` /
/// `fir.declare` model their effect; declaring a variable allocates nothing
/// at run time).
///
/// Anything that writes, allocates or frees real memory (e.g. `fir.store`,
/// `fir.alloca`, `hlfir.assign`), and any op whose effects are unknown (e.g. a
/// `fir.call`), is rejected.
static bool isSafeToRunPerTeam(mlir::Operation *op) {
  std::optional<llvm::SmallVector<mlir::MemoryEffects::EffectInstance>>
      effects = mlir::getEffectsRecursively(op);
  // No memory effect interface somewhere in `op`: effects are unknown.
  if (!effects)
    return false;

  return llvm::all_of(
      *effects, [](const mlir::MemoryEffects::EffectInstance &effect) {
        return mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()) ||
               mlir::isa<fir::DebuggingResource>(effect.getResource());
      });
}

/// Decides whether a `teams distribute parallel do` nest may be emitted for
/// \p loop inside the user-written target region \p targetOp.
///
/// OpenMP requires that "if a teams construct is nested within a target
/// construct, that target construct must contain no statements, declarations
/// or directives outside of the teams construct". Flang's semantic checks
/// enforce this on user source, but this pass runs after semantics and
/// creates the teams construct itself, so it must check on its own.
///
/// The rule is stated on source statements while this pass sees FIR ops, and
/// some ops in the target come from the loop header rather than from separate
/// statements. For example, for
///
///   !$omp target map(tofrom: a) map(to: n)
///   do concurrent (i = 1:n)
///     a(i) = a(i) + 1
///   end do
///   !$omp end target
///
/// the target region contains, besides the loop, the `hlfir.declare` ops of
/// the mapped variables and a `fir.load` of `n` computing the upper bound.
/// What actually matters is the consequence of the rule: code that is inside
/// the target but outside the teams region is executed once per team. Reads
/// and pure computation are harmless when repeated, writes and calls are not
/// (`x = x + 1` would run once per team). Therefore a teams nest is emitted
/// only if:
///
///   1. \p loop is directly in the body of \p targetOp (not inside an
///      `fir.if`, a sequential loop, or another OpenMP construct within the
///      target; a teams region there would be illegal);
///   2. \p loop is the only `fir.do_concurrent` in \p targetOp (a target may
///      contain at most one teams region);
///   3. every other op in the target's body exists only to serve the loop or
///      to declare a mapped variable: it is a declaration, or each of its
///      users is the loop, a declaration, or another such op (unused ops must
///      be constants). In addition, each such op must be safe to run once per
///      team (`isSafeToRunPerTeam`).
///
/// Condition 3 makes the rewrite equivalent, in source terms, to turning a
/// target region that contains only the `do concurrent` into one that
/// contains only a `teams distribute parallel do` construct, which is legal
/// OpenMP. After the conversion, the code left in the target body before
/// omp.teams is only declarations, constants and shape computations, the
/// same as lowering produces for the hand-written construct (the loop's own
/// computations are re-evaluated in omp.parallel and the originals erased;
/// see cloneTargetBodyValuesIntoParallel).
static bool canEmitTeamsInTarget(fir::DoConcurrentOp loop,
                                 mlir::omp::TargetOp targetOp) {
  if (loop->getParentOp() != targetOp.getOperation())
    return false;

  unsigned numLoops = 0;
  targetOp.getRegion().walk([&](fir::DoConcurrentOp) { ++numLoops; });
  if (numLoops != 1)
    return false;

  // Condition 3: every other op only serves the loop or declares a mapped
  // variable. The body is visited in reverse so that, when an op is examined,
  // all of its users in the body have already been classified.
  if (!targetOp.getRegion().hasOneBlock())
    return false;
  mlir::Block &body = targetOp.getRegion().front();
  llvm::SmallPtrSet<mlir::Operation *, 16> servesLoop;
  for (mlir::Operation &op : llvm::reverse(body)) {
    if (&op == loop.getOperation() || mlir::isa<mlir::omp::TerminatorOp>(op))
      continue;

    // Even code that only serves the loop is executed once per team, so it
    // must not write, allocate or call anything.
    if (!isSafeToRunPerTeam(&op))
      return false;

    // Declarations of variables (typically the mapped variables) are allowed
    // regardless of their users: they only give a name and, for arrays, a
    // shape to storage that already exists. Lowering emits them in the
    // target body for legal `!$omp target` + `!$omp teams ...` code too.
    if (mlir::isa<hlfir::DeclareOp, fir::DeclareOp>(op)) {
      servesLoop.insert(&op);
      continue;
    }

    // Unused ops are only allowed if they are constants. Anything else
    // without users (an unused read, or an unused computation that would make
    // an otherwise unrelated read look like it serves the loop) did not come
    // from the loop header or a declaration.
    if (op.use_empty()) {
      if (!op.hasTrait<mlir::OpTrait::ConstantLike>())
        return false;
      servesLoop.insert(&op);
      continue;
    }

    // Every user must be the loop (a bound, or a value used in the loop
    // body), a declaration, or another op that itself only serves the loop.
    // For example, a `fir.load` of a mapped variable feeding the loop's upper
    // bound passes, and so does a `fir.load` of an extent feeding the shape
    // of an explicit-shape dummy array's declaration.
    for (mlir::Operation *user : op.getUsers()) {
      mlir::Operation *userInBody = body.findAncestorOpInBlock(*user);
      if (userInBody == loop.getOperation() ||
          (userInBody && servesLoop.contains(userInBody)))
        continue;
      return false;
    }
    servesLoop.insert(&op);
  }

  return true;
}

class DoConcurrentConversion
    : public mlir::OpConversionPattern<fir::DoConcurrentOp> {
private:
  struct TargetDeclareShapeCreationInfo {
    // Note: We use `std::vector` (rather than `llvm::SmallVector` as usual) to
    // interface more easily `ShapeShiftOp::getOrigins()` which returns
    // `std::vector`.
    std::vector<mlir::Value> startIndices;
    std::vector<mlir::Value> extents;

    TargetDeclareShapeCreationInfo(mlir::Value liveIn) {
      mlir::Value shape = nullptr;
      mlir::Operation *liveInDefiningOp = liveIn.getDefiningOp();
      auto declareOp =
          mlir::dyn_cast_if_present<hlfir::DeclareOp>(liveInDefiningOp);

      if (declareOp != nullptr)
        shape = declareOp.getShape();

      if (!shape)
        return;

      mlir::Operation *shapeDefOp = shape.getDefiningOp();

      if (auto shapeOp = mlir::dyn_cast_if_present<fir::ShapeOp>(shapeDefOp)) {
        extents.assign(shapeOp.getExtents().begin(),
                       shapeOp.getExtents().end());
      } else if (auto shapeShiftOp =
                     mlir::dyn_cast_if_present<fir::ShapeShiftOp>(shapeDefOp)) {
        startIndices = shapeShiftOp.getOrigins();
        extents = shapeShiftOp.getExtents();
      } else if (auto shiftOp =
                     mlir::dyn_cast_if_present<fir::ShiftOp>(shapeDefOp)) {
        startIndices.assign(shiftOp.getOrigins().begin(),
                            shiftOp.getOrigins().end());
      }
    }

    bool isShapedValue() const {
      return !extents.empty() || !startIndices.empty();
    }
    bool isShapeShiftedValue() const {
      return !startIndices.empty() && !extents.empty();
    }
    bool isShiftedValue() const {
      return !startIndices.empty() && extents.empty();
    }
  };

  using LiveInShapeInfoMap =
      llvm::DenseMap<mlir::Value, TargetDeclareShapeCreationInfo>;

public:
  using mlir::OpConversionPattern<fir::DoConcurrentOp>::OpConversionPattern;

  DoConcurrentConversion(
      mlir::MLIRContext *context, bool mapToDevice,
      llvm::DenseSet<fir::DoConcurrentOp> &concurrentLoopsToSkip,
      mlir::SymbolTable &moduleSymbolTable)
      : OpConversionPattern(context), mapToDevice(mapToDevice),
        concurrentLoopsToSkip(concurrentLoopsToSkip),
        moduleSymbolTable(moduleSymbolTable) {}

  mlir::LogicalResult
  matchAndRewrite(fir::DoConcurrentOp doLoop, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    looputils::InductionVariableInfos ivInfos;
    auto loop = mlir::cast<fir::DoConcurrentLoopOp>(
        doLoop.getRegion().back().getTerminator());

    auto indVars = loop.getLoopInductionVars();
    assert(indVars.has_value());

    for (mlir::Value indVar : *indVars)
      ivInfos.emplace_back(loop, indVar);

    llvm::SmallVector<mlir::Value> loopNestLiveIns;
    looputils::collectLoopLiveIns(loop, loopNestLiveIns);
    assert(!loopNestLiveIns.empty());

    llvm::SetVector<mlir::Value> locals;
    looputils::collectLoopLocalValues(loop, locals);

    // We do not want to map "loop-local" values to the device through
    // `omp.map.info` ops. Therefore, we remove them from the list of live-ins.
    loopNestLiveIns.erase(llvm::remove_if(loopNestLiveIns,
                                          [&](mlir::Value liveIn) {
                                            return locals.contains(liveIn);
                                          }),
                          loopNestLiveIns.end());

    // Decide once, for this particular loop, which construct nest to emit.
    // The code below never tests `ctx` (or the `mapToDevice` option) directly;
    // it calls the predicates defined next to `EmissionContext`
    // (createsTargetRegion, remapsLiveIns, distributesAcrossTeams,
    // privatizesOnParallel), each of which derives one property of the nest
    // from `ctx`.
    const EmissionContext ctx = computeEmissionContext(doLoop);

    mlir::omp::TargetOp targetOp;
    mlir::omp::LoopNestOperands loopNestClauseOps;

    mlir::IRMapping mapper;

    if (createsTargetRegion(ctx)) {
      mlir::ModuleOp module = doLoop->getParentOfType<mlir::ModuleOp>();
      bool isTargetDevice =
          llvm::cast<mlir::omp::OffloadModuleInterface>(*module)
              .getIsTargetDevice();

      mlir::omp::TargetExtOperands targetClauseOps;
      genLoopNestClauseOps(doLoop.getLoc(), rewriter, loop, loopNestClauseOps,
                           isTargetDevice ? nullptr : &targetClauseOps);

      LiveInShapeInfoMap liveInShapeInfoMap;
      fir::FirOpBuilder builder(
          rewriter,
          fir::getKindMapping(doLoop->getParentOfType<mlir::ModuleOp>()));

      for (mlir::Value liveIn : loopNestLiveIns) {
        bool isReductionVar = llvm::find(loop.getReduceVars(), liveIn) !=
                              loop.getReduceVars().end();
        targetClauseOps.mapVars.push_back(
            genMapInfoOpForLiveIn(builder, liveIn, isReductionVar));
        liveInShapeInfoMap.insert(
            {liveIn, TargetDeclareShapeCreationInfo(liveIn)});
      }

      targetClauseOps.kernelType = mlir::omp::TargetExecModeAttr::get(
          rewriter.getContext(), mlir::omp::TargetExecMode::spmd);
      targetOp =
          genTargetOp(doLoop.getLoc(), rewriter, mapper, loopNestLiveIns,
                      targetClauseOps, loopNestClauseOps, liveInShapeInfoMap);
      targetOp.setCombined(true);
    }

    // The teams construct is created whenever iterations are distributed
    // across teams: either directly inside the new target region created
    // above (NewDeviceTarget), or inside the user's existing target region
    // (TeamsInUserTarget).
    if (distributesAcrossTeams(ctx)) {
      auto teamsOp = genTeamsOp(rewriter, loop, mapper, ctx);
      teamsOp.setCombined(true);

      // Mark the user's target `omp.combined`, exactly as lowering does for
      // user code written as `!$omp target` + nested `!$omp teams ...` (it
      // marks such a target combined even when its kernel type stays
      // `generic`). The marker is what makes the target "capture" the nested
      // teams construct (`ComposableOpInterface::findCapturedOp`), and
      // translation to LLVM IR uses the captured op to set up the kernel's
      // launch configuration. Without the marker the kernel is treated as one
      // without a teams construct: it still computes the right result, but
      // the host requests a single team (the runtime's "no teams" value, -1),
      // so the whole loop runs on one compute unit. With it, the runtime
      // chooses the number of teams (e.g. 624 = 104 CUs x 6 on gfx90a for a
      // Generic-SPMD kernel).
      //
      // The user's kernel type is kept (normally `generic`): the loop bounds
      // are computed inside the region, so they cannot be host-evaluated as an
      // `spmd` kernel would require, and OpenMPOpt promotes the generic kernel
      // to Generic-SPMD at -O1 and above anyway.
      if (ctx == EmissionContext::TeamsInUserTarget)
        doLoop->getParentOfType<mlir::omp::TargetOp>().setCombined(true);
    }

    // Collects info about LOCAL_INIT variables that need per-iteration copy
    // inlining. Populated by genPrivatizers(), which is called from
    // genParallelOp or from genWsLoopOp depending on `privatizesOnParallel`.
    // Consumed below after block arg remapping to inline the copy_region into
    // omp.loop_nest.
    llvm::SmallVector<looputils::LocalInitCopyInfo> localInitCopies;

    mlir::omp::ParallelOp parallelOp =
        genParallelOp(rewriter, loop, ivInfos, mapper, localInitCopies, ctx);

    // Evaluate the computations the loop depends on inside omp.parallel, as
    // lowering does (see cloneTargetBodyValuesIntoParallel).
    llvm::SmallVector<mlir::Operation *> clonedTargetBodyOps;
    if (ctx == EmissionContext::TeamsInUserTarget)
      clonedTargetBodyOps = cloneTargetBodyValuesIntoParallel(
          rewriter, doLoop, doLoop->getParentOfType<mlir::omp::TargetOp>(),
          mapper);

    // Only set as composite when part of `distribute parallel do`, and only set
    // as combined when part of `parallel do`.
    parallelOp.setComposite(distributesAcrossTeams(ctx));
    parallelOp.setCombined(!distributesAcrossTeams(ctx));

    // When a new target region is created, the loop nest clause operands were
    // already generated above (their bounds must be host-evaluated and remapped
    // into the target region). Otherwise generate them here, in place.
    if (!createsTargetRegion(ctx)) {
      genLoopNestClauseOps(doLoop.getLoc(), rewriter, loop, loopNestClauseOps);
      // Bounds computed in the user's target body may have been cloned into
      // the parallel region (see cloneTargetBodyValuesIntoParallel); use the
      // clones.
      for (auto *bounds :
           {&loopNestClauseOps.loopLowerBounds,
            &loopNestClauseOps.loopUpperBounds, &loopNestClauseOps.loopSteps})
        for (mlir::Value &bound : *bounds)
          bound = mapper.lookupOrDefault(bound);
    }

    for (mlir::Value local : locals)
      looputils::localizeLoopLocalValue(local, parallelOp.getRegion(),
                                        rewriter);

    if (distributesAcrossTeams(ctx))
      genDistributeOp(doLoop.getLoc(), rewriter).setComposite(/*val=*/true);

    auto [loopNestOp, wsLoopOp] = genWsLoopOp(
        rewriter, loop, mapper, loopNestClauseOps, localInitCopies, ctx);

    // `local` region arguments are transferred/cloned from the `do concurrent`
    // loop to the loopnest op when the region is cloned above. Instead, these
    // region arguments should be on the op that owns the corresponding clause:
    //   - privates:   omp.parallel (privatizesOnParallel) or omp.wsloop;
    //   - reductions: omp.wsloop in both cases.
    if (privatizesOnParallel(ctx)) {
      for (auto [parallelArg, loopNestArg] : llvm::zip_equal(
               parallelOp.getRegion().getArguments(),
               loopNestOp.getRegion().getArguments().slice(
                   loop.getLocalOperandsStart(), loop.getNumLocalOperands())))
        rewriter.replaceAllUsesWith(loopNestArg, parallelArg);

      for (auto [wsloopArg, loopNestArg] : llvm::zip_equal(
               wsLoopOp.getRegion().getArguments(),
               loopNestOp.getRegion().getArguments().slice(
                   loop.getReduceOperandsStart(), loop.getNumReduceOperands())))
        rewriter.replaceAllUsesWith(loopNestArg, wsloopArg);
    } else {
      for (auto [wsloopArg, loopNestArg] :
           llvm::zip_equal(wsLoopOp.getRegion().getArguments(),
                           loopNestOp.getRegion().getArguments().drop_front(
                               loopNestClauseOps.loopLowerBounds.size())))
        rewriter.replaceAllUsesWith(loopNestArg, wsloopArg);
    }

    for (unsigned i = 0;
         i < loop.getLocalVars().size() + loop.getReduceVars().size(); ++i)
      loopNestOp.getRegion().eraseArgument(
          loopNestClauseOps.loopLowerBounds.size());

    // Inline copy_region for LOCAL_INIT variables at the top of the
    // omp.loop_nest body. This must happen after the block arg remapping above,
    // because we need the final (remapped) private block args to map the
    // copy_region's arguments to.
    //
    // For each LOCAL_INIT variable, we resolve the privateVarIdx to the actual
    // block arg on whichever op holds the private clause (omp.parallel for
    // device, omp.wsloop for host), then inline the copy_region ops at the
    // start of the loop_nest body. This gives per-iteration initialization:
    //
    //   omp.loop_nest (%iv) = ... {
    //     // inlined copy: re-init private from original every iteration
    //     %v = fir.load %originalVar
    //     fir.store %v to %privateVar
    //     // ... rest of loop body ...
    //   }
    if (!localInitCopies.empty()) {
      for (auto &info : localInitCopies) {
        // Resolve the private block arg. Privates are on omp.parallel when
        // `privatizesOnParallel(ctx)` holds (distribute parallel do nest), and
        // on omp.wsloop otherwise (parallel do nest).
        //
        // We use BlockArgOpenMPOpInterface::getPrivateBlockArgs() rather than
        // raw getRegion().getArgument(idx) because the block arg layout
        // depends on which clause types are present. The layout order is:
        //   hasDeviceAddr, hostEval, inReduction, map, private, reduction, ...
        // Using getPrivateBlockArgs() correctly accounts for any preceding
        // clause block args, making the indexing robust if new clause types
        // are added to the op in the future.
        mlir::Operation *privArgOwner = privatizesOnParallel(ctx)
                                            ? parallelOp.getOperation()
                                            : wsLoopOp.getOperation();
        auto blockArgIface =
            mlir::cast<mlir::omp::BlockArgOpenMPOpInterface>(privArgOwner);
        mlir::Value privateBlockArg =
            blockArgIface.getPrivateBlockArgs()[info.privateVarIdx];

        if (mlir::failed(looputils::inlineCopyRegionToLoopBody(
                rewriter, info.localizer.getCopyRegion(), info.originalVar,
                privateBlockArg, loopNestOp.getRegion().front())))
          return mlir::failure();
      }
    }

    rewriter.setInsertionPoint(doLoop);
    fir::FirOpBuilder builder(
        rewriter,
        fir::getKindMapping(doLoop->getParentOfType<mlir::ModuleOp>()));

    // Collect iteration variable(s) allocations so that we can move them
    // outside the `fir.do_concurrent` wrapper (before erasing it).
    //
    // Not for TeamsInUserTarget: there the nearest allocation block is the
    // user's target body, i.e. the code before omp.teams, which must only
    // contain declarations, constants and shape computations (see
    // canEmitTeamsInTarget). The loop body uses the copies of these ops
    // created inside omp.parallel (genLoopNestIndVarAllocs), so the originals
    // are simply erased together with the wrapper below. Values defined in
    // the wrapper cannot be used outside of it, so nothing else refers to
    // them.
    if (ctx != EmissionContext::TeamsInUserTarget) {
      llvm::SmallVector<mlir::Operation *> opsToMove;
      for (mlir::Operation &op : llvm::drop_end(doLoop))
        opsToMove.push_back(&op);

      mlir::Block *allocBlock = builder.getAllocaBlock();

      for (mlir::Operation *op : llvm::reverse(opsToMove)) {
        rewriter.moveOpBefore(op, allocBlock, allocBlock->begin());
      }
    }

    // Mark `unordered` loops that are not perfectly nested to be skipped from
    // the legality check of the `ConversionTarget` since we are not interested
    // in mapping them to OpenMP.
    loopNestOp->walk([&](fir::DoConcurrentOp doLoop) {
      concurrentLoopsToSkip.insert(doLoop);
    });

    rewriter.eraseOp(doLoop);

    // The originals of the target-body computations that were re-evaluated in
    // omp.parallel were only used by the (now erased) loop. Erase them, users
    // before definitions, so that the code before omp.teams is the same as
    // lowering produces for `!$omp target` + `!$omp teams distribute parallel
    // do`. An original that still has users is kept.
    for (mlir::Operation *op : llvm::reverse(clonedTargetBodyOps))
      if (op->use_empty())
        rewriter.eraseOp(op);

    return mlir::success();
  }

private:
  /// For a loop converted inside the user's target region (TeamsInUserTarget),
  /// clone into the just-created omp.parallel region every computation from
  /// the target's body whose result the loop uses (typically the loop bounds,
  /// e.g. a `fir.load` of a mapped variable), and map the originals to the
  /// clones in \p mapper. The insertion point must be inside the parallel
  /// region, before the omp.distribute op is created.
  ///
  /// Why: this reproduces where lowering places such computations. For user
  /// code written as `!$omp target` / `!$omp teams` / `!$omp distribute
  /// parallel do`, lowering evaluates the loop bounds inside omp.parallel,
  /// right before omp.distribute (it cannot place statements between `target`
  /// and `teams`, which OpenMP forbids). Leaving the computations in the
  /// target body and using their results as bounds of the composite
  /// `distribute parallel do` nest was observed to miscompile: with
  /// unoptimized front-end output that OpenMPOpt promotes to Generic-SPMD
  /// during device LTO (flang's default when no -O flag is given), a bound
  /// loaded in the target body reached the loop with a wrong value (writing 40
  /// of 10000 elements, or a GPU memory access fault). The root cause further
  /// down the pipeline was not identified; the `parallel do` fallback with the
  /// same kind of bound is not affected.
  ///
  /// Cloning is safe: `canEmitTeamsInTarget` only accepts target bodies
  /// whose ops, apart from the loop, compute values without writing anything,
  /// such as constants, conversions and loads of mapped variables. Every
  /// thread that runs the cloned ops computes the same values, so the results
  /// do not change. Declarations are not cloned; they only rename mapped
  /// variables and are used as is.
  ///
  /// Returns the original ops that were cloned, in program order. Once the
  /// `fir.do_concurrent` op is erased they normally have no users left, and
  /// the caller erases them, so that only declarations, constants and shape
  /// computations remain in the target body before omp.teams.
  llvm::SmallVector<mlir::Operation *> cloneTargetBodyValuesIntoParallel(
      mlir::ConversionPatternRewriter &rewriter, fir::DoConcurrentOp doLoop,
      mlir::omp::TargetOp targetOp, mlir::IRMapping &mapper) const {
    // An op is cloned if it is directly in the target's body (not nested in
    // another op of the body), is not the loop itself, and is not a
    // declaration.
    auto isClonableBodyOp = [&](mlir::Operation *op) {
      return op && op->getParentOp() == targetOp.getOperation() &&
             op != doLoop.getOperation() &&
             !mlir::isa<hlfir::DeclareOp, fir::DeclareOp>(op);
    };

    // Collect, transitively, the body ops whose results the loop (its bounds
    // or its body) uses.
    llvm::SetVector<mlir::Operation *> toClone;
    llvm::SmallVector<mlir::Value> worklist;
    mlir::visitUsedValuesDefinedAbove(
        doLoop->getRegions(),
        [&](mlir::OpOperand *operand) { worklist.push_back(operand->get()); });
    while (!worklist.empty()) {
      mlir::Operation *def = worklist.pop_back_val().getDefiningOp();
      if (!isClonableBodyOp(def) || !toClone.insert(def))
        continue;
      for (mlir::Value operand : def->getOperands())
        worklist.push_back(operand);
    }
    // Clone in original program order so that operands precede their users.
    llvm::SmallVector<mlir::Operation *> ordered(toClone.begin(),
                                                 toClone.end());
    llvm::sort(ordered, [](mlir::Operation *a, mlir::Operation *b) {
      return a->isBeforeInBlock(b);
    });
    for (mlir::Operation *op : ordered)
      rewriter.clone(*op, mapper);
    return ordered;
  }

  mlir::omp::ParallelOp genParallelOp(
      mlir::ConversionPatternRewriter &rewriter, fir::DoConcurrentLoopOp loop,
      looputils::InductionVariableInfos &ivInfos, mlir::IRMapping &mapper,
      llvm::SmallVectorImpl<looputils::LocalInitCopyInfo> &localInitCopies,
      EmissionContext ctx) const {
    mlir::omp::ParallelOperands parallelOps;

    // In a `distribute parallel do` nest, private variables are attached to
    // omp.parallel (not omp.wsloop); see `privatizesOnParallel`.
    // genPrivatizers populates localInitCopies for any LOCAL_INIT variables
    // encountered.
    if (privatizesOnParallel(ctx))
      genPrivatizers(rewriter, mapper, loop, parallelOps, localInitCopies, ctx);

    mlir::Location loc = loop.getLoc();
    auto parallelOp = mlir::omp::ParallelOp::create(rewriter, loc, parallelOps);
    Fortran::common::openmp::EntryBlockArgs parallelArgs;
    parallelArgs.privVars = parallelOps.privateVars;
    Fortran::common::openmp::genEntryBlock(rewriter, parallelArgs,
                                           parallelOp.getRegion());
    rewriter.setInsertionPoint(mlir::omp::TerminatorOp::create(rewriter, loc));

    genLoopNestIndVarAllocs(rewriter, ivInfos, mapper);
    return parallelOp;
  }

  void genLoopNestIndVarAllocs(mlir::ConversionPatternRewriter &rewriter,
                               looputils::InductionVariableInfos &ivInfos,
                               mlir::IRMapping &mapper) const {

    for (auto &indVarInfo : ivInfos)
      genInductionVariableAlloc(rewriter, indVarInfo.iterVarMemDef, mapper);
  }

  mlir::Operation *
  genInductionVariableAlloc(mlir::ConversionPatternRewriter &rewriter,
                            mlir::Operation *indVarMemDef,
                            mlir::IRMapping &mapper) const {
    assert(
        indVarMemDef != nullptr &&
        "Induction variable memdef is expected to have a defining operation.");

    llvm::SmallSetVector<mlir::Operation *, 2> indVarDeclareAndAlloc;
    for (auto operand : indVarMemDef->getOperands())
      indVarDeclareAndAlloc.insert(operand.getDefiningOp());
    indVarDeclareAndAlloc.insert(indVarMemDef);

    mlir::Operation *result;
    for (mlir::Operation *opToClone : indVarDeclareAndAlloc)
      result = rewriter.clone(*opToClone, mapper);

    return result;
  }

  void genLoopNestClauseOps(
      mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
      fir::DoConcurrentLoopOp loop,
      mlir::omp::LoopNestOperands &loopNestClauseOps,
      mlir::omp::TargetExtOperands *targetClauseOps = nullptr) const {
    assert(loopNestClauseOps.loopLowerBounds.empty() &&
           "Loop nest bounds were already emitted!");

    auto populateBounds = [](mlir::Value var,
                             llvm::SmallVectorImpl<mlir::Value> &bounds) {
      bounds.push_back(var.getDefiningOp()->getResult(0));
    };

    auto hostEvalCapture = [&](mlir::Value var,
                               llvm::SmallVectorImpl<mlir::Value> &bounds) {
      populateBounds(var, bounds);

      // Ensure that loop-nest bounds are evaluated in the host and forwarded to
      // the nested omp constructs when we map to the device.
      if (targetClauseOps)
        targetClauseOps->hostEvalVars.push_back(var);
    };

    for (auto [lb, ub, st] : llvm::zip_equal(
             loop.getLowerBound(), loop.getUpperBound(), loop.getStep())) {
      hostEvalCapture(lb, loopNestClauseOps.loopLowerBounds);
      hostEvalCapture(ub, loopNestClauseOps.loopUpperBounds);
      hostEvalCapture(st, loopNestClauseOps.loopSteps);
    }

    loopNestClauseOps.loopInclusive = rewriter.getUnitAttr();
    loopNestClauseOps.collapseNumLoops =
        rewriter.getI64IntegerAttr(loopNestClauseOps.loopLowerBounds.size());
  }

  std::pair<mlir::omp::LoopNestOp, mlir::omp::WsloopOp> genWsLoopOp(
      mlir::ConversionPatternRewriter &rewriter, fir::DoConcurrentLoopOp loop,
      mlir::IRMapping &mapper, const mlir::omp::LoopNestOperands &clauseOps,
      llvm::SmallVectorImpl<looputils::LocalInitCopyInfo> &localInitCopies,
      EmissionContext ctx) const {
    mlir::omp::WsloopOperands wsloopClauseOps;
    // In a plain `parallel do` nest, private variables are attached to
    // omp.wsloop (not omp.parallel); see `privatizesOnParallel`.
    // genPrivatizers populates localInitCopies for any LOCAL_INIT variables
    // encountered.
    if (!privatizesOnParallel(ctx))
      genPrivatizers(rewriter, mapper, loop, wsloopClauseOps, localInitCopies,
                     ctx);

    genReductions(rewriter, mapper, loop, wsloopClauseOps, ctx);

    auto wsloopOp =
        mlir::omp::WsloopOp::create(rewriter, loop.getLoc(), wsloopClauseOps);
    // The worksharing loop is part of the composite `distribute parallel do`
    // construct exactly when iterations are also distributed across teams.
    wsloopOp.setComposite(distributesAcrossTeams(ctx));

    Fortran::common::openmp::EntryBlockArgs wsloopArgs;
    wsloopArgs.privVars = wsloopClauseOps.privateVars;
    wsloopArgs.reductionVars = wsloopClauseOps.reductionVars;
    Fortran::common::openmp::genEntryBlock(rewriter, wsloopArgs,
                                           wsloopOp.getRegion());

    auto loopNestOp =
        mlir::omp::LoopNestOp::create(rewriter, loop.getLoc(), clauseOps);

    // Clone the loop's body inside the loop nest construct using the
    // mapped values.
    rewriter.cloneRegionBefore(loop.getRegion(), loopNestOp.getRegion(),
                               loopNestOp.getRegion().begin(), mapper);

    rewriter.setInsertionPointToEnd(&loopNestOp.getRegion().back());
    mlir::omp::YieldOp::create(rewriter, loop->getLoc());

    return {loopNestOp, wsloopOp};
  }

  void genBoundsOps(fir::FirOpBuilder &builder, mlir::Value liveIn,
                    mlir::Value rawAddr,
                    llvm::SmallVectorImpl<mlir::Value> &boundsOps) const {
    fir::ExtendedValue extVal =
        hlfir::translateToExtendedValue(rawAddr.getLoc(), builder,
                                        hlfir::Entity{liveIn},
                                        /*contiguousHint=*/
                                        true)
            .first;
    fir::factory::AddrAndBoundsInfo info = fir::factory::getDataOperandBaseAddr(
        builder, rawAddr, /*isOptional=*/false, rawAddr.getLoc());
    boundsOps = fir::factory::genImplicitBoundsOps<mlir::omp::MapBoundsOp,
                                                   mlir::omp::MapBoundsType>(
        builder, info, extVal,
        /*dataExvIsAssumedSize=*/false, rawAddr.getLoc());
  }

  static bool recordHasAllocatableMember(fir::RecordType recordType) {
    for (auto [fieldName, fieldType] : recordType.getTypeList()) {
      if (fir::isPointerType(fir::unwrapRefType(fieldType)))
        continue;

      if (fir::isAllocatableType(fieldType))
        return true;

      if (auto nestedRec = mlir::dyn_cast<fir::RecordType>(
              fir::getFortranElementType(fieldType)))
        if (recordHasAllocatableMember(nestedRec))
          return true;
    }

    return false;
  }

  mlir::omp::MapInfoOp
  genMapInfoOpForLiveIn(fir::FirOpBuilder &builder, mlir::Value liveIn,
                        bool isReductionVar = false) const {
    mlir::Value rawAddr = liveIn;
    llvm::StringRef name;

    mlir::Operation *liveInDefiningOp = liveIn.getDefiningOp();
    auto declareOp =
        mlir::dyn_cast_if_present<hlfir::DeclareOp>(liveInDefiningOp);

    if (declareOp != nullptr) {
      // Use the raw address to avoid unboxing `fir.box` values whenever
      // possible. Put differently, if we have access to the direct value memory
      // reference/address, we use it.
      rawAddr = declareOp.getOriginalBase();
      name = declareOp.getUniqName();
    }

    if (!llvm::isa<mlir::omp::PointerLikeType>(rawAddr.getType())) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(liveInDefiningOp);
      auto copyVal = builder.createTemporary(liveIn.getLoc(), liveIn.getType());
      builder.createStoreWithConvert(copyVal.getLoc(), liveIn, copyVal);
      rawAddr = copyVal;
    }

    mlir::Type liveInType = liveIn.getType();
    mlir::Type eleType = liveInType;
    if (auto refType = mlir::dyn_cast<fir::ReferenceType>(liveInType))
      eleType = refType.getElementType();

    mlir::omp::ClauseMapFlags mapFlag = mlir::omp::ClauseMapFlags::implicit;
    mlir::omp::VariableCaptureKind captureKind =
        mlir::omp::VariableCaptureKind::ByRef;

    if (isReductionVar) {
      mapFlag |= mlir::omp::ClauseMapFlags::to;
      mapFlag |= mlir::omp::ClauseMapFlags::from;
    } else if (fir::isa_trivial(eleType) || fir::isa_char(eleType)) {
      captureKind = mlir::omp::VariableCaptureKind::ByCopy;
    } else if (!fir::isa_builtin_cptr_type(eleType)) {
      mapFlag |= mlir::omp::ClauseMapFlags::to;
      mapFlag |= mlir::omp::ClauseMapFlags::from;
    }

    llvm::SmallVector<mlir::Value> boundsOps;
    genBoundsOps(builder, liveIn, rawAddr, boundsOps);

    fir::RecordType recordType = mlir::dyn_cast<fir::RecordType>(
        fir::getDerivedType(fir::unwrapRefType(eleType)));

    bool requiresImplicitMapper =
        recordType && recordHasAllocatableMember(recordType);

    mlir::FlatSymbolRefAttr mapperId;
    if (requiresImplicitMapper) {
      std::string mapperIdName =
          Fortran::utils::openmp::getCanonicalDefaultDeclareMapperName(
              recordType);
      mapperId = Fortran::utils::openmp::getOrGenImplicitDefaultDeclareMapper(
          builder, liveIn.getLoc(), recordType, mapperIdName);
    }

    return Fortran::utils::openmp::createMapInfoOp(
        builder, liveIn.getLoc(), rawAddr,
        /*varPtrPtr=*/{}, name.str(), boundsOps,
        /*members=*/{},
        /*membersIndex=*/mlir::ArrayAttr{}, mapFlag, captureKind,
        rawAddr.getType(), /*partialMap=*/false, mapperId);
  }

  mlir::omp::TargetOp
  genTargetOp(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
              mlir::IRMapping &mapper, llvm::ArrayRef<mlir::Value> mappedVars,
              mlir::omp::TargetExtOperands &clauseOps,
              mlir::omp::LoopNestOperands &loopNestClauseOps,
              const LiveInShapeInfoMap &liveInShapeInfoMap) const {
    auto targetOp = mlir::omp::TargetOp::create(rewriter, loc, clauseOps);
    auto argIface = llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*targetOp);

    mlir::Region &region = targetOp.getRegion();

    llvm::SmallVector<mlir::Type> regionArgTypes;
    llvm::SmallVector<mlir::Location> regionArgLocs;

    for (auto var : llvm::concat<const mlir::Value>(clauseOps.hostEvalVars,
                                                    clauseOps.mapVars)) {
      regionArgTypes.push_back(var.getType());
      regionArgLocs.push_back(var.getLoc());
    }

    rewriter.createBlock(&region, {}, regionArgTypes, regionArgLocs);
    fir::FirOpBuilder builder(
        rewriter,
        fir::getKindMapping(targetOp->getParentOfType<mlir::ModuleOp>()));

    // Within the loop, it is possible that we discover other values that need
    // to be mapped to the target region (the shape info values for arrays, for
    // example). Therefore, the map block args might be extended and resized.
    // Hence, we invoke `argIface.getMapBlockArgs()` every iteration to make
    // sure we access the proper vector of data.
    int idx = 0;
    for (auto [mapInfoOp, mappedVar] :
         llvm::zip_equal(clauseOps.mapVars, mappedVars)) {
      auto miOp = mlir::cast<mlir::omp::MapInfoOp>(mapInfoOp.getDefiningOp());
      hlfir::DeclareOp liveInDeclare =
          genLiveInDeclare(builder, targetOp, argIface.getMapBlockArgs()[idx],
                           miOp, liveInShapeInfoMap.at(mappedVar));
      ++idx;

      // If `mappedVar.getDefiningOp()` is a `fir::BoxAddrOp`, we probably
      // need to "unpack" the box by getting the defining op of it's value.
      // However, we did not hit this case in reality yet so leaving it as a
      // todo for now.
      if (mlir::isa<fir::BoxAddrOp>(mappedVar.getDefiningOp()))
        TODO(mappedVar.getLoc(),
             "Mapped variabled defined by `BoxAddrOp` are not supported yet");

      auto mapHostValueToDevice = [&](mlir::Value hostValue,
                                      mlir::Value deviceValue) {
        if (!llvm::isa<mlir::omp::PointerLikeType>(hostValue.getType()))
          mapper.map(hostValue,
                     builder.loadIfRef(hostValue.getLoc(), deviceValue));
        else
          mapper.map(hostValue, deviceValue);
      };

      // Only one result of a host `hlfir.declare` is collected as a live-in,
      // but the loop may use both, so map both regardless of which one it is.
      if (auto origDeclareOp = mlir::dyn_cast_if_present<hlfir::DeclareOp>(
              mappedVar.getDefiningOp())) {
        mapHostValueToDevice(origDeclareOp.getOriginalBase(),
                             liveInDeclare.getOriginalBase());
        mapHostValueToDevice(origDeclareOp.getBase(), liveInDeclare.getBase());
      } else {
        mapHostValueToDevice(mappedVar, liveInDeclare.getOriginalBase());
      }
    }

    for (auto [arg, hostEval] : llvm::zip_equal(argIface.getHostEvalBlockArgs(),
                                                clauseOps.hostEvalVars))
      mapper.map(hostEval, arg);

    for (unsigned i = 0; i < loopNestClauseOps.loopLowerBounds.size(); ++i) {
      loopNestClauseOps.loopLowerBounds[i] =
          mapper.lookup(loopNestClauseOps.loopLowerBounds[i]);
      loopNestClauseOps.loopUpperBounds[i] =
          mapper.lookup(loopNestClauseOps.loopUpperBounds[i]);
      loopNestClauseOps.loopSteps[i] =
          mapper.lookup(loopNestClauseOps.loopSteps[i]);
    }

    // Check if cloning the bounds introduced any dependency on the outer
    // region. If so, then either clone them as well if they are
    // MemoryEffectFree, or else copy them to a new temporary and add them to
    // the map and block_argument lists and replace their uses with the new
    // temporary.
    Fortran::utils::openmp::cloneOrMapRegionOutsiders(builder, targetOp);
    rewriter.setInsertionPoint(
        mlir::omp::TerminatorOp::create(rewriter, targetOp.getLoc()));

    return targetOp;
  }

  hlfir::DeclareOp genLiveInDeclare(
      fir::FirOpBuilder &builder, mlir::omp::TargetOp targetOp,
      mlir::Value liveInArg, mlir::omp::MapInfoOp liveInMapInfoOp,
      const TargetDeclareShapeCreationInfo &targetShapeCreationInfo) const {
    mlir::Type liveInType = liveInArg.getType();
    std::string liveInName = liveInMapInfoOp.getName().has_value()
                                 ? liveInMapInfoOp.getName().value().str()
                                 : std::string("");
    if (fir::isa_ref_type(liveInType))
      liveInType = fir::unwrapRefType(liveInType);

    // Constants are left for `cloneOrMapRegionOutsiders` to rematerialize
    // inside the target region instead of being mapped through a temporary.
    auto mapShapeValue = [&](mlir::Value value, llvm::StringRef kind,
                             size_t dim) -> mlir::Value {
      if (isIntConstant(value))
        return value;
      return Fortran::utils::openmp::mapTemporaryValue(
          builder, targetOp, value,
          liveInName + "." + kind.str() + ".dim" + std::to_string(dim));
    };

    if (targetShapeCreationInfo.isShiftedValue()) {
      // A `fir.shift` is only valid on a box value. Box live-ins are mapped
      // through a temporary, so load the box before re-declaring it.
      assert(fir::isa_ref_type(liveInArg.getType()) &&
             "expected a box live-in mapped through a temporary");
      mlir::Value box =
          fir::LoadOp::create(builder, liveInArg.getLoc(), liveInArg);
      llvm::SmallVector<mlir::Value> origins;
      for (auto [idx, startIndex] :
           llvm::enumerate(targetShapeCreationInfo.startIndices))
        origins.push_back(mapShapeValue(startIndex, "start_idx", idx));
      auto shiftType =
          fir::ShiftType::get(builder.getContext(), origins.size());
      mlir::Value shift =
          fir::ShiftOp::create(builder, liveInArg.getLoc(), shiftType, origins);
      return hlfir::DeclareOp::create(builder, liveInArg.getLoc(), box,
                                      liveInName, shift);
    }

    mlir::Value shape = [&]() -> mlir::Value {
      if (!targetShapeCreationInfo.isShapedValue())
        return {};

      if (targetShapeCreationInfo.isShapeShiftedValue()) {
        llvm::SmallVector<mlir::Value> shapeShiftOperands;

        size_t shapeIdx = 0;
        for (auto [startIndex, extent] :
             llvm::zip_equal(targetShapeCreationInfo.startIndices,
                             targetShapeCreationInfo.extents)) {
          shapeShiftOperands.push_back(
              mapShapeValue(startIndex, "start_idx", shapeIdx));
          shapeShiftOperands.push_back(
              mapShapeValue(extent, "extent", shapeIdx));
          ++shapeIdx;
        }

        auto shapeShiftType = fir::ShapeShiftType::get(
            builder.getContext(), shapeShiftOperands.size() / 2);
        return fir::ShapeShiftOp::create(builder, liveInArg.getLoc(),
                                         shapeShiftType, shapeShiftOperands);
      }

      llvm::SmallVector<mlir::Value> shapeOperands;
      size_t shapeIdx = 0;
      for (auto extent : targetShapeCreationInfo.extents) {
        shapeOperands.push_back(mapShapeValue(extent, "extent", shapeIdx));
        ++shapeIdx;
      }

      return fir::ShapeOp::create(builder, liveInArg.getLoc(), shapeOperands);
    }();

    return hlfir::DeclareOp::create(builder, liveInArg.getLoc(), liveInArg,
                                    liveInName, shape);
  }

  mlir::omp::TeamsOp genTeamsOp(mlir::ConversionPatternRewriter &rewriter,
                                fir::DoConcurrentLoopOp loop,
                                mlir::IRMapping &mapper,
                                EmissionContext ctx) const {
    mlir::omp::TeamsOperands teamsOps;
    genReductions(rewriter, mapper, loop, teamsOps, ctx);

    mlir::Location loc = loop.getLoc();
    auto teamsOp = mlir::omp::TeamsOp::create(rewriter, loc, teamsOps);
    Fortran::common::openmp::EntryBlockArgs teamsArgs;
    teamsArgs.reductionVars = teamsOps.reductionVars;
    Fortran::common::openmp::genEntryBlock(rewriter, teamsArgs,
                                           teamsOp.getRegion());

    rewriter.setInsertionPoint(mlir::omp::TerminatorOp::create(rewriter, loc));

    for (auto [loopVar, teamsArg] : llvm::zip_equal(
             loop.getReduceVars(), teamsOp.getRegion().getArguments())) {
      mapper.map(loopVar, teamsArg);
    }

    return teamsOp;
  }

  mlir::omp::DistributeOp
  genDistributeOp(mlir::Location loc,
                  mlir::ConversionPatternRewriter &rewriter) const {
    auto distOp = mlir::omp::DistributeOp::create(
        rewriter, loc, /*clauses=*/mlir::omp::DistributeOperands{});

    rewriter.createBlock(&distOp.getRegion());
    return distOp;
  }

  void cloneFIRRegionToOMP(mlir::ConversionPatternRewriter &rewriter,
                           mlir::Region &firRegion,
                           mlir::Region &ompRegion) const {
    if (!firRegion.empty()) {
      rewriter.cloneRegionBefore(firRegion, ompRegion, ompRegion.begin());
      auto firYield =
          mlir::cast<fir::YieldOp>(ompRegion.back().getTerminator());
      rewriter.setInsertionPoint(firYield);
      mlir::omp::YieldOp::create(rewriter, firYield.getLoc(),
                                 firYield.getOperands());
      rewriter.eraseOp(firYield);
    }
  }

  /// Generate bodies of OpenMP privatizers by cloning the bodies of FIR
  /// privatizers.
  ///
  /// For both LOCAL and LOCAL_INIT variables, this creates an omp.private op
  /// with DataSharingClauseType::Private and clones the init and dealloc
  /// regions. The copy_region is NOT cloned into omp.private for LOCAL_INIT
  /// because omp.private copy_region fires once per thread (firstprivate
  /// semantics), but LOCAL_INIT requires per-iteration initialization. Instead,
  /// LOCAL_INIT copy info is recorded in \p localInitCopies for later inlining
  /// into the omp.loop_nest body by the caller.
  ///
  /// \param [in] rewriter - used to drive IR generation for privatizers.
  /// \param [in] mapper - value mapping from FIR to OpenMP constructs.
  /// \param [in] loop - FIR loop to convert its localizers.
  ///
  /// \param [out] privateClauseOps - OpenMP privatizers to gen their bodies.
  /// \param [out] localInitCopies - info for LOCAL_INIT variables that need
  ///   per-iteration copy inlining. Populated here, consumed after block arg
  ///   remapping in matchAndRewrite().
  /// \param [in] ctx - the emission context of the loop; decides whether
  ///   \p loop's local variables must be remapped to device values.
  void genPrivatizers(
      mlir::ConversionPatternRewriter &rewriter, mlir::IRMapping &mapper,
      fir::DoConcurrentLoopOp loop,
      mlir::omp::PrivateClauseOps &privateClauseOps,
      llvm::SmallVectorImpl<looputils::LocalInitCopyInfo> &localInitCopies,
      EmissionContext ctx) const {
    // For `local` and `local_init` operands, emit corresponding `private`
    // clauses and attach these clauses to the workshare loop (host) or
    // parallel op (device).
    if (!loop.getLocalVars().empty()) {
      // Track the index into privateClauseOps.privateVars so we can later
      // resolve it to the wsloop/parallel block arg for LOCAL_INIT inlining.
      unsigned privateVarIdx = 0;

      for (auto [var, sym, arg] : llvm::zip_equal(
               loop.getLocalVars(),
               loop.getLocalSymsAttr().getAsRange<mlir::SymbolRefAttr>(),
               loop.getRegionLocalArgs())) {
        auto localizer = moduleSymbolTable.lookup<fir::LocalitySpecifierOp>(
            sym.getLeafReference());

        // Both LOCAL and LOCAL_INIT get the same omp.private {type = private}
        // privatizer with init + dealloc regions (no copy_region). This gives
        // each thread its own storage.
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(localizer);

        auto privatizer = mlir::omp::PrivateClauseOp::create(
            rewriter, localizer.getLoc(), sym.getLeafReference().str() + ".omp",
            /*sym_visibility=*/nullptr, localizer.getTypeAttr().getValue(),
            mlir::omp::DataSharingClauseType::Private);

        cloneFIRRegionToOMP(rewriter, localizer.getInitRegion(),
                            privatizer.getInitRegion());
        cloneFIRRegionToOMP(rewriter, localizer.getDeallocRegion(),
                            privatizer.getDeallocRegion());

        moduleSymbolTable.insert(privatizer);

        // When the loop was moved into a new (IsolatedFromAbove) target
        // region, use the mapped (device-side) value. Otherwise use the raw
        // value, which is still visible because omp.parallel is not
        // IsolatedFromAbove.
        mlir::Value actualVar = remapsLiveIns(ctx) ? mapper.lookup(var) : var;
        privateClauseOps.privateVars.push_back(actualVar);
        privateClauseOps.privateSyms.push_back(
            mlir::SymbolRefAttr::get(privatizer));

        // For LOCAL_INIT, record the info needed for per-iteration copy
        // inlining. The copy_region on the fir.local op will be inlined at
        // the top of the omp.loop_nest body later in matchAndRewrite(), after
        // block arg remapping has established the final SSA values.
        if (localizer.getLocalitySpecifierType() ==
            fir::LocalitySpecifierType::LocalInit) {
          localInitCopies.push_back({/*localizer=*/localizer,
                                     /*originalVar=*/actualVar,
                                     /*privateVarIdx=*/privateVarIdx});
        }

        ++privateVarIdx;
      }
    }
  }

  /// Generate `omp.declare_reduction` ops for \p loop's `reduce` operands
  /// (cloning the bodies of the corresponding `fir.declare_reduction` ops, or
  /// reusing an already converted one) and append the reduction clause
  /// operands to \p reductionClauseOps. \p ctx decides whether the reduced
  /// variables must be remapped to device values first.
  void genReductions(mlir::ConversionPatternRewriter &rewriter,
                     mlir::IRMapping &mapper, fir::DoConcurrentLoopOp loop,
                     mlir::omp::ReductionClauseOps &reductionClauseOps,
                     EmissionContext ctx) const {
    if (!loop.getReduceVars().empty()) {
      for (auto [var, byRef, sym, arg] : llvm::zip_equal(
               loop.getReduceVars(), loop.getReduceByrefAttr().asArrayRef(),
               loop.getReduceSymsAttr().getAsRange<mlir::SymbolRefAttr>(),
               loop.getRegionReduceArgs())) {
        auto firReducer = moduleSymbolTable.lookup<fir::DeclareReductionOp>(
            sym.getLeafReference());

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(firReducer);
        std::string ompReducerName = sym.getLeafReference().str() + ".omp";

        auto ompReducer =
            moduleSymbolTable.lookup<mlir::omp::DeclareReductionOp>(
                rewriter.getStringAttr(ompReducerName));

        if (!ompReducer) {
          ompReducer = mlir::omp::DeclareReductionOp::create(
              rewriter, firReducer.getLoc(), ompReducerName,
              /*sym_visibility=*/nullptr, firReducer.getTypeAttr().getValue(),
              firReducer.getByrefElementTypeAttr());

          cloneFIRRegionToOMP(rewriter, firReducer.getAllocRegion(),
                              ompReducer.getAllocRegion());
          cloneFIRRegionToOMP(rewriter, firReducer.getInitializerRegion(),
                              ompReducer.getInitializerRegion());
          cloneFIRRegionToOMP(rewriter, firReducer.getReductionRegion(),
                              ompReducer.getReductionRegion());
          cloneFIRRegionToOMP(rewriter, firReducer.getAtomicReductionRegion(),
                              ompReducer.getAtomicReductionRegion());
          cloneFIRRegionToOMP(rewriter, firReducer.getCleanupRegion(),
                              ompReducer.getCleanupRegion());
          moduleSymbolTable.insert(ompReducer);
        }

        // Which SSA value is reduced into depends on what was created before
        // this clause:
        //   - Host / ParallelInUserTarget: nothing remaps `var`, use it as is.
        //   - NewDeviceTarget, teams clause: `var` was mapped to the new
        //     target's block argument by genTargetOp.
        //   - NewDeviceTarget / TeamsInUserTarget, wsloop clause: genTeamsOp
        //     mapped `var` to the omp.teams reduction block argument, so the
        //     worksharing loop reduces into the per-team private copy, which
        //     the teams reduction then combines.
        // `lookupOrDefault` covers all cases; when the loop was moved into a
        // new target region a mapping must exist.
        assert((!remapsLiveIns(ctx) || mapper.contains(var)) &&
               "reduction variable not mapped into the new target region");
        reductionClauseOps.reductionVars.push_back(mapper.lookupOrDefault(var));
        reductionClauseOps.reductionByref.push_back(byRef);
        reductionClauseOps.reductionSyms.push_back(
            mlir::SymbolRefAttr::get(ompReducer));
      }
    }
  }

  /// Decide which construct nest to emit for \p doLoop (see
  /// `EmissionContext`).
  ///
  /// This is the single place that interprets the user's
  /// `-fdo-concurrent-to-openmp` request. The decision also depends on where
  /// \p doLoop is:
  ///
  ///   =host                                     -> Host
  ///   =device, not inside any omp.target        -> NewDeviceTarget
  ///   =device, inside a user omp.target, and a teams nest may be emitted
  ///            there (canEmitTeamsInTarget)     -> TeamsInUserTarget
  ///   =device, inside a user omp.target otherwise -> ParallelInUserTarget
  ///
  /// With =host, a loop inside a user target keeps getting the Host nest,
  /// which is also legal there (same as before this distinction existed).
  EmissionContext computeEmissionContext(fir::DoConcurrentOp doLoop) const {
    if (!mapToDevice)
      return EmissionContext::Host;

    auto enclosingTarget = doLoop->getParentOfType<mlir::omp::TargetOp>();
    if (!enclosingTarget)
      return EmissionContext::NewDeviceTarget;

    return canEmitTeamsInTarget(doLoop, enclosingTarget)
               ? EmissionContext::TeamsInUserTarget
               : EmissionContext::ParallelInUserTarget;
  }

  /// The user's request: `-fdo-concurrent-to-openmp=device` (true) or `=host`
  /// (false). Only read by `computeEmissionContext`; all code generation is
  /// driven by the resulting `EmissionContext`.
  bool mapToDevice;
  llvm::DenseSet<fir::DoConcurrentOp> &concurrentLoopsToSkip;
  mlir::SymbolTable &moduleSymbolTable;
};

/// A listener that forwards notifyOperationErased to the given callback.
struct CallbackListener : public mlir::RewriterBase::Listener {
  CallbackListener(std::function<void(mlir::Operation *op)> onOperationErased)
      : onOperationErased(onOperationErased) {}

  void notifyOperationErased(mlir::Operation *op) override {
    onOperationErased(op);
  }

  std::function<void(mlir::Operation *op)> onOperationErased;
};

class DoConcurrentConversionPass
    : public flangomp::impl::DoConcurrentConversionPassBase<
          DoConcurrentConversionPass> {
public:
  DoConcurrentConversionPass() = default;

  DoConcurrentConversionPass(
      const flangomp::DoConcurrentConversionPassOptions &options)
      : DoConcurrentConversionPassBase(options) {}

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *context = &getContext();
    mlir::SymbolTable moduleSymbolTable(module);

    if (mapTo != flangomp::DoConcurrentMappingKind::DCMK_Host &&
        mapTo != flangomp::DoConcurrentMappingKind::DCMK_Device) {
      mlir::emitWarning(mlir::UnknownLoc::get(context),
                        "DoConcurrentConversionPass: invalid `map-to` value. "
                        "Valid values are: `host` or `device`");
      return;
    }

    llvm::DenseSet<fir::DoConcurrentOp> concurrentLoopsToSkip;
    CallbackListener callbackListener([&](mlir::Operation *op) {
      if (auto loop = mlir::dyn_cast<fir::DoConcurrentOp>(op))
        concurrentLoopsToSkip.erase(loop);
    });
    mlir::RewritePatternSet patterns(context);
    patterns.insert<DoConcurrentConversion>(
        context, mapTo == flangomp::DoConcurrentMappingKind::DCMK_Device,
        concurrentLoopsToSkip, moduleSymbolTable);
    mlir::ConversionTarget target(*context);
    target.addDynamicallyLegalOp<fir::DoConcurrentOp>(
        [&](fir::DoConcurrentOp op) {
          return concurrentLoopsToSkip.contains(op);
        });
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });

    mlir::ConversionConfig config;
    config.allowPatternRollback = false;
    config.listener = &callbackListener;
    if (mlir::failed(mlir::applyFullConversion(module, target,
                                               std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
flangomp::createDoConcurrentConversionPass(bool mapToDevice) {
  DoConcurrentConversionPassOptions options;
  options.mapTo = mapToDevice ? flangomp::DoConcurrentMappingKind::DCMK_Device
                              : flangomp::DoConcurrentMappingKind::DCMK_Host;

  return std::make_unique<DoConcurrentConversionPass>(options);
}
