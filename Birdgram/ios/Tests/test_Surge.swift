import Surge
import Bubo

func M(_ rows: Int, _ columns: Int, _ repeatedValue: Float) -> Matrix<Float> {
  return Matrix(rows: rows, columns: columns, repeatedValue: repeatedValue)
}

// Ensure fix is in place for https://github.com/mattt/Surge/issues/92
test("Surge: X*Y doesn't crash on empty input") { name in
  testEqual(name, M(1,2,.nan) * M(2,0,.nan), M(1,0,.nan))
  testEqual(name, M(0,2,.nan) * M(2,3,.nan), M(0,3,.nan))
  testEqual(name, M(1,0,.nan) * M(0,3,.nan), M(1,3,0)) // Mimic numpy (nonempty zero matrix)
}

test("Surge: x*Y doesn't crash on empty input") { name in
  testEqual(name, 3 * M(2,0,.nan), M(2,0,.nan))
  testEqual(name, 3 * M(0,3,.nan), M(0,3,.nan))
  testEqual(name, 3 * M(0,0,.nan), M(0,0,.nan))
}

test("Surge: X+Y doesn't crash on empty input") { name in
  testEqual(name, M(2,0,.nan) + M(2,0,.nan), M(2,0,.nan))
  testEqual(name, M(0,2,.nan) + M(0,2,.nan), M(0,2,.nan))
}

test("Surge: X-Y doesn't crash on empty input") { name in
  testEqual(name, M(2,0,.nan) - M(2,0,.nan), M(2,0,.nan))
  testEqual(name, M(0,2,.nan) - M(0,2,.nan), M(0,2,.nan))
}

test("Surge: inv(X) doesn't crash on empty input") { name in
  testEqual(name, inv(M(1,1,1)), M(1,1,1))
  testEqual(name, inv(M(0,0,0)), M(0,0,0)) // Mimic numpy
}

test("Surge: transpose(X) doesn't crash on empty input") { name in
  testEqual(name, transpose(M(1,1,0)), M(1,1,0))
  testEqual(name, transpose(M(0,0,0)), M(0,0,0))
}

test("Surge: det(X) doesn't crash on empty input") { name in
  testEqual(name, det(M(1,1,1)), 1)
  testEqual(name, det(M(0,0,0)), nil) // (Blas prints warning, but Surge handles return code)
}

test("Surge: subscript(rows: / columns: / rows:columns:)") { name in
  let rows: [[Float]] = [
    [ 0,  1,  2,  3,  4],
    [ 5,  6,  7,  8,  9],
    [10, 11, 12, 13, 14],
  ]
  let M = Matrix(rows)
  let columns: [[Float]] = [
    [ 0,  5, 10],
    [ 1,  6, 11],
    [ 2,  7, 12],
    [ 3,  8, 13],
    [ 4,  9, 14],
  ]
  testEqual(name, M[rows: 0..<3], Matrix(rows[0..<3]))
  testEqual(name, M[rows: 1..<3], Matrix(rows[1..<3]))
  testEqual(name, M[rows: 1..<2], Matrix(rows[1..<2]))
  testEqual(name, M[rows: 1..<1], Matrix(rows[1..<1], rows: 0, columns: M.columns)) // 0xN not 0x0
  testEqual(name, M[columns: 0..<5], Matrix(columns[0..<5]).T)
  testEqual(name, M[columns: 0..<4], Matrix(columns[0..<4]).T)
  testEqual(name, M[columns: 1..<2], Matrix(columns[1..<2]).T)
  testEqual(name, M[columns: 2..<2], Matrix(columns[2..<2], rows: 0, columns: M.rows).T) // Nx0 not 0x0
}
