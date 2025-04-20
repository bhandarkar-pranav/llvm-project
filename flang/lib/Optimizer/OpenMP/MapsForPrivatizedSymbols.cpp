//===- MapsForPrivatizedSymbols.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// An OpenMP dialect related pass for FIR/HLFIR which creates MapInfoOp
/// instances for certain privatized symbols.
/// For example, if an allocatable variable is used in a private clause attached
/// to a omp.target op, then the allocatable variable's descriptor will be
/// needed on the device (e.g. GPU). This descriptor needs to be separately
/// mapped onto the device. This pass creates the necessary omp.map.info ops for
/// this.
//===----------------------------------------------------------------------===//
// TODO:
// 1. Before adding omp.map.info, check if we already have an omp.map.info for
// the variable in question.
// 2. Generalize this for more than just omp.target ops.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Support/Debug.h"
#include <type_traits>

#define DEBUG_TYPE "omp-maps-for-privatized-symbols"
#define PDBGS() (llvm::dbgs() << "[" << DEBUG_TYPE << "]: ")
namespace flangomp {
#define GEN_PASS_DEF_MAPSFORPRIVATIZEDSYMBOLSPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

using namespace mlir;

namespace {
class MapsForPrivatizedSymbolsPass
    : public flangomp::impl::MapsForPrivatizedSymbolsPassBase<
          MapsForPrivatizedSymbolsPass> {

  int createMapInfo(Location loc, Value var,
                     fir::FirOpBuilder &builder,
                     llvm::SmallVectorImpl<omp::MapInfoOp> &mapInfoOps) {
    uint64_t mapTypeTo = static_cast<
        std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
        llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO);
    Operation *definingOp = var.getDefiningOp();
    int64_t numMapsGenerated = 0;

    Value varPtr = var;
    // We want the first result of the hlfir.declare op because our goal
    // is to map the descriptor (fir.box or fir.boxchar) and the first
    // result for hlfir.declare is the descriptor if a the symbol being
    // declared needs a descriptor.
    // Some types are boxed immediately before privatization. These have other
    // operations in between the privatization and the declaration. It is safe
    // to use var directly here because they will be boxed anyway.
    if (auto declOp = llvm::dyn_cast_if_present<hlfir::DeclareOp>(definingOp))
      varPtr = declOp.getBase();

    // If the varPtr is a BoxCharType, we should create a MapInfo for the underlying
    // data pointer inside the BoxChar. Since we are generating a new MapInfoOp
    // for the privatized BoxChar, we do not check if the underlying pointer has
    // already been mapped.
    if (mlir::isa<fir::BoxCharType>(varPtr.getType())) {
      // We need to map the !fir.char<k> wrapped by the !fir.boxchar<k>
      // Step 1. From !fir.boxchar, get the !fir.ref<fir.char<k>
      fir::BoxCharType boxType = mlir::cast<fir::BoxCharType>(varPtr.getType());
      mlir::Type refType = builder.getRefType(boxType.getEleTy());
      mlir::Type lenType = builder.getCharacterLengthType();
      auto unboxed = builder.create<fir::UnboxCharOp>(loc, refType, lenType, varPtr);
      mlir::Value charRef = unboxed.getResult(0);

      // Step 2. Create the MapInfoOp to map charRef.
      mlir::omp::MapInfoOp mapInfoForCharRef = builder.create<omp::MapInfoOp>(
          loc, charRef.getType(), charRef,
          TypeAttr::get(llvm::cast<omp::PointerLikeType>(charRef.getType())
                            .getElementType()),
          builder.getIntegerAttr(builder.getIntegerType(64, /*isSigned=*/false),
                                 mapTypeTo),
          builder.getAttr<omp::VariableCaptureKindAttr>(
              omp::VariableCaptureKind::ByRef),
          /*varPtrPtr=*/Value{},
          /*members=*/SmallVector<Value>{},
          /*member_index=*/mlir::ArrayAttr{},
          /*bounds=*/ValueRange{},
          /*mapperId=*/mlir::FlatSymbolRefAttr(), /*name=*/StringAttr(),
          builder.getBoolAttr(false));
      mapInfoOps.push_back(mapInfoForCharRef);
      numMapsGenerated += 1;
      LLVM_DEBUG(PDBGS() << "createMap for charRef: " << mapInfoForCharRef
                 << "\n");
    }
    // If we do not have a reference to a descriptor but the descriptor itself,
    // then we need to store that on the stack so that we can map the
    // address of the descriptor.
    if (mlir::isa<fir::BaseBoxType>(varPtr.getType()) ||
        mlir::isa<fir::BoxCharType>(varPtr.getType())) {
      OpBuilder::InsertPoint savedInsPoint = builder.saveInsertionPoint();
      mlir::Block *allocaBlock = builder.getAllocaBlock();
      assert(allocaBlock && "No allocablock  found for a funcOp");
      builder.setInsertionPointToStart(allocaBlock);
      auto alloca = builder.create<fir::AllocaOp>(loc, varPtr.getType());
      builder.restoreInsertionPoint(savedInsPoint);
      builder.create<fir::StoreOp>(loc, varPtr, alloca);
      varPtr = alloca;
    }
    assert(mlir::isa<omp::PointerLikeType>(varPtr.getType()) &&
           "Dealing with a varPtr that is not a PointerLikeType");

    // Figure out the bounds because knowing the bounds will help the subsequent
    // MapInfoFinalizationPass map the underlying data of the descriptor.
    llvm::SmallVector<mlir::Value> boundsOps;
    if (needsBoundsOps(varPtr))
      genBoundsOps(builder, varPtr, boundsOps);

    mlir::omp::MapInfoOp mapInfoOp =  builder.create<omp::MapInfoOp>(
        loc, varPtr.getType(), varPtr,
        TypeAttr::get(llvm::cast<omp::PointerLikeType>(varPtr.getType())
                          .getElementType()),
        builder.getIntegerAttr(builder.getIntegerType(64, /*isSigned=*/false),
                               mapTypeTo),
        builder.getAttr<omp::VariableCaptureKindAttr>(
            omp::VariableCaptureKind::ByRef),
        /*varPtrPtr=*/Value{},
        /*members=*/SmallVector<Value>{},
        /*member_index=*/mlir::ArrayAttr{},
        /*bounds=*/boundsOps.empty() ? SmallVector<Value>{} : boundsOps,
        /*mapperId=*/mlir::FlatSymbolRefAttr(), /*name=*/StringAttr(),
        builder.getBoolAttr(false));
    mapInfoOps.push_back(mapInfoOp);
    numMapsGenerated += 1;
    LLVM_DEBUG(PDBGS() << "MapsForPrivatizedSymbolsPass created ->\n"
               << mapInfoOp << "\n");
    return numMapsGenerated;
  }
  void addMapInfoOp(omp::TargetOp targetOp, omp::MapInfoOp mapInfoOp) {
    auto argIface = llvm::cast<omp::BlockArgOpenMPOpInterface>(*targetOp);
    unsigned insertIndex =
        argIface.getMapBlockArgsStart() + argIface.numMapBlockArgs();
    targetOp.getMapVarsMutable().append(ValueRange{mapInfoOp});
    targetOp.getRegion().insertArgument(insertIndex, mapInfoOp.getType(),
                                        mapInfoOp.getLoc());
  }
  void addMapInfoOps(omp::TargetOp targetOp,
                     llvm::SmallVectorImpl<omp::MapInfoOp> &mapInfoOps) {
    for (auto mapInfoOp : mapInfoOps)
      addMapInfoOp(targetOp, mapInfoOp);
  }
  void runOnOperation() override {
    ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
    fir::KindMapping kindMap = fir::getKindMapping(module);
    fir::FirOpBuilder builder{module, std::move(kindMap)};
    llvm::DenseMap<Operation *, llvm::SmallVector<omp::MapInfoOp, 4>>
        mapInfoOpsForTarget;

    getOperation()->walk([&](omp::TargetOp targetOp) {
      if (targetOp.getPrivateVars().empty())
        return;
      OperandRange privVars = targetOp.getPrivateVars();
      llvm::SmallVector<int64_t> privVarMapIdx;

      std::optional<ArrayAttr> privSyms = targetOp.getPrivateSyms();
      SmallVector<omp::MapInfoOp, 4> mapInfoOps;
      for (auto [privVar, privSym] : llvm::zip_equal(privVars, *privSyms)) {

        SymbolRefAttr privatizerName = llvm::cast<SymbolRefAttr>(privSym);
        omp::PrivateClauseOp privatizer =
            SymbolTable::lookupNearestSymbolFrom<omp::PrivateClauseOp>(
                targetOp, privatizerName);
        if (!privatizer.needsMap()) {
          privVarMapIdx.push_back(-1);
          continue;
        }

        int numMapInfosAlready = targetOp.getMapVars().size() +
                                     mapInfoOps.size();
        builder.setInsertionPoint(targetOp);
        Location loc = targetOp.getLoc();
        int numMapInfosAdded = createMapInfo(loc, privVar, builder, mapInfoOps);
        privVarMapIdx.push_back(numMapInfosAlready + numMapInfosAdded - 1);


      }
      if (!mapInfoOps.empty()) {
        mapInfoOpsForTarget.insert({targetOp.getOperation(), mapInfoOps});
        targetOp.setPrivateMapsAttr(
            mlir::DenseI64ArrayAttr::get(targetOp.getContext(), privVarMapIdx));
      }
    });
    if (!mapInfoOpsForTarget.empty()) {
      for (auto &[targetOp, mapInfoOps] : mapInfoOpsForTarget) {
        addMapInfoOps(static_cast<omp::TargetOp>(targetOp), mapInfoOps);
      }
    }
  }
  // As the name suggests, this function examines var to determine if
  // it has dynamic size. If true, this pass'll have to extract these
  // bounds from descriptor of var and add the bounds to the resultant
  // MapInfoOp.
  bool needsBoundsOps(mlir::Value var) {
    assert(mlir::isa<omp::PointerLikeType>(var.getType()) &&
           "needsBoundsOps can deal only with pointer types");
    mlir::Type t = fir::unwrapRefType(var.getType());
    // t could be a box, so look inside the box
    auto innerType = fir::dyn_cast_ptrOrBoxEleTy(t);
    if (innerType)
      return fir::hasDynamicSize(innerType);
    return fir::hasDynamicSize(t);
  }
  void genBoundsOps(fir::FirOpBuilder &builder, mlir::Value var,
                    llvm::SmallVector<mlir::Value> &boundsOps) {
    if (!fir::isBoxAddress(var.getType()))
      return;

    mlir::Location loc = var.getLoc();
    mlir::Type idxTy = builder.getIndexType();
    mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
    mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
    mlir::Type boundTy = builder.getType<omp::MapBoundsType>();
    mlir::Value box = builder.create<fir::LoadOp>(loc, var);
    unsigned int rank = 0;
    rank = fir::getBoxRank(fir::unwrapRefType(var.getType()));

    auto genBoundsOp = [&](mlir::Value lb, mlir::Value extent, mlir::Value stride, mlir::Value start_idx) -> mlir::omp::MapBoundsOp {
      mlir::Value ub = builder.create<mlir::arith::SubIOp>(loc, extent, one);
      return builder.create<omp::MapBoundsOp>(
          loc, boundTy, /*lower_bound=*/lb,
          /*upper_bound=*/ub, /*extent=*/extent, /*stride=*/stride,
          /*stride_in_bytes = */ true, /*start_idx=*/start_idx);
    };
    if (rank == 0) {
      // What else can have rank 0? If nothing else can, then shouldn't we
      // assert instead of return?
      if (!fir::factory::CharacterExprHelper::isCharacterScalar(var.getType()))
        return;
      LLVM_DEBUG(PDBGS() << "is Character\n");
      mlir::Value extent =
          fir::factory::CharacterExprHelper{builder, loc}.readLengthFromBox(
              box);
      mlir::Value boundsOp =
          genBoundsOp(zero, extent, /*stride=*/one, /*start_idx=*/zero);
      LLVM_DEBUG(PDBGS() << "Created BoundsOp " << boundsOp << "\n");
      boundsOps.push_back(boundsOp);
    } else {
      for (unsigned int i = 0; i < rank; ++i) {
        mlir::Value dimNo = builder.createIntegerConstant(loc, idxTy, i);
        auto dimInfo = builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy,
                                                      box, dimNo);
        auto normalizedLB = builder.create<mlir::arith::ConstantOp>(
            loc, idxTy, builder.getIntegerAttr(idxTy, 0));

        mlir::Value boundsOp =
            genBoundsOp(normalizedLB, dimInfo.getExtent(),
                        dimInfo.getByteStride(), dimInfo.getLowerBound());
        LLVM_DEBUG(PDBGS() << "Created BoundsOp " << boundsOp << "\n");
        boundsOps.push_back(boundsOp);
      }
    }
  }
};
} // namespace
