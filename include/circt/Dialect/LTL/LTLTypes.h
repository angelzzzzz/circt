//===- LTLTypes.h - LTL type declarations -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LTL_LTLTYPES_H
#define CIRCT_DIALECT_LTL_LTLTYPES_H

#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

// Include generated types.
#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/LTL/LTLTypes.h.inc"

#endif // CIRCT_DIALECT_LTL_LTLTYPES_H
