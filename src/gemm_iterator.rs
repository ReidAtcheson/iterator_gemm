pub fn gemm(_m : usize, _n : usize, _p : usize, a : &[f64], lda : usize, b : &[f64], ldb : usize, c : &mut [f64], ldc : usize) -> () {
    for (ci,ai) in c.chunks_exact_mut(ldc).zip(a.chunks_exact(lda)){
        for (aik,bk) in ai.iter().zip(b.chunks_exact(ldb)){
            for (cij,bkj) in ci.iter_mut().zip(bk.iter()){
                *cij += (*aik) * (*bkj);
            }
        }
    }
}
