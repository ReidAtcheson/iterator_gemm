pub fn gemm(m : usize, n : usize, p : usize, a : &[f64], lda : usize, b : &[f64], ldb : usize, c : &mut [f64], ldc : usize) -> () {

    for i in 0..m{
        for k in 0..p{
            for j in 0..n{
                let aik=a[k+lda*i];
                let bkj=b[j+ldb*k];
                let ref mut cij=c[j+ldc*i];
                *cij += aik*bkj;
            }
        }
    }

}
