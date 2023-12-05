// RUN: circt-opt %s --arc-vectorize-states | FileCheck %s

// CHECK-LABEL: hw.module @Foo
hw.module @Foo(in %clock: !seq.clock, in %en: i1, in %inA: i3, in %inB: i3) {
  %4 = arc.state @FooMux(%en, %21753, %4) clock %clock lat 1 : (i1, i3, i3) -> i3
  %5 = arc.state @FooMux(%en, %21754, %5) clock %clock lat 1 : (i1, i3, i3) -> i3
  %7 = arc.state @FooMux(%en, %21756, %7) clock %clock lat 1 : (i1, i3, i3) -> i3
  %12 = arc.state @FooMux(%en, %91, %12) clock %clock lat 1 : (i1, i3, i3) -> i3
  %15 = arc.state @FooMux(%en, %93, %15) clock %clock lat 1 : (i1, i3, i3) -> i3
  %16 = arc.state @FooMux(%en, %94, %16) clock %clock lat 1 : (i1, i3, i3) -> i3

  %21753 = comb.xor %200, %inA : i3
  %21754 = comb.xor %201, %inA : i3
  %21756 = comb.xor %202, %inA : i3

  %91 = comb.add %100, %inB : i3
  %93 = comb.add %101, %inB : i3
  %94 = comb.add %102, %inB : i3

  %100 = comb.mul %12, %inA : i3
  %101 = comb.mul %15, %inA : i3
  %102 = comb.sub %16, %inA : i3

  %200 = comb.and %4, %inB : i3
  %201 = comb.and %5, %inB : i3
  %202 = comb.and %7, %inB : i3
}

arc.define @FooMux(%arg0: i1, %arg1: i3, %arg2: i3) -> i3 {
  %0 = comb.mux bin %arg0, %arg1, %arg2 : i3
  arc.output %0 : i3
}
