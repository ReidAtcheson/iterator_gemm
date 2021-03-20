


pub fn gemm(m : usize, n : usize, p : usize, a : &[f64], lda : usize, b : &[f64], ldb : usize, c : &mut [f64], ldc : usize) -> () {

    let ap=a.as_ptr();
    let bp=b.as_ptr();
    let mut cp=c.as_mut_ptr();


    let ldai = lda as isize;
    let ldbi = ldb as isize;
    let ldci = ldc as isize;

    for i in 0..(m as isize){
        for k in 0..(p as isize){
            for j in 0..(n as isize){
                let ik = k+ldai*i;
                let kj = j+ldbi*k;
                let ij = j+ldci*i;
                unsafe{
                    let aik = *(ap.offset(ik));
                    let bkj = *(bp.offset(kj));
                    let mut cij = cp.offset(ij);
                    *cij += aik*bkj;
                }
            }
        }
    }

}
