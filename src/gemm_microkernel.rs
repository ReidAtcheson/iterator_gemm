use ndarray::{ArrayViewMut2,ArrayView2,s};


pub fn gemm(m : usize, n : usize, p : usize, a : &[f64], _lda : usize, b : &[f64], _ldb : usize, c : &mut [f64], _ldc : usize) -> () {

    //Wrap data slices into ArrayViews from ndarray crate.
    //This gives good iterators over subarrays
    let ablk_big = ArrayView2::<f64>::from_shape((m,p),a).unwrap();
    let bblk_big = ArrayView2::<f64>::from_shape((p,m),b).unwrap();
    let mut cblk_big = ArrayViewMut2::<f64>::from_shape((m,n),c).unwrap();


    const BI : usize = 64;
    const BJ : usize = 64;
    const BK : usize = 64;

    const BQ : isize = 4;
    const BR : isize = 64;
    const BS : isize = 4;

    let mut ab = [0.0;BI*BK];
    let mut bb = [0.0;BJ*BK];
    let mut cb = [0.0;BI*BJ];
    let ap=ab.as_ptr();
    let bp=bb.as_ptr();
    let mut cp=cb.as_mut_ptr();


    for i in (0..m).step_by(BI){
        for j in (0..n).step_by(BJ){
            for k in (0..p).step_by(BK){
                let iend=std::cmp::min(m,i+BI);
                let jend=std::cmp::min(n,j+BJ);
                let kend=std::cmp::min(p,k+BK);
                let ablk = ablk_big.slice(s![i..iend,k..kend]);
                let bblk = bblk_big.slice(s![k..kend,j..jend]);
                let mut cblk = cblk_big.slice_mut(s![i..iend,j..jend]);

                //Copy data from full matrices into smaller workspaces
                for (x,y) in ablk.iter().zip(ab.iter_mut()){
                    *y=*x;
                }
                for (x,y) in bblk.iter().zip(bb.iter_mut()){
                    *y=*x;
                }
                for (x,y) in cblk.iter().zip(cb.iter_mut()){
                    *y=*x;
                }

                let q=(iend-i) as isize;
                let r=(jend-j) as isize;
                let s=(kend-k) as isize;

                for yi in ((0 as isize)..q).step_by(BQ as usize){
                    for yk in ((0 as isize)..s).step_by(BS as usize){
                        for yj in ((0 as isize)..r).step_by(BR as usize){
                            let yiend = std::cmp::min(q,yi+BQ);
                            let ykend = std::cmp::min(s,yk+BS);
                            let yjend = std::cmp::min(r,yj+BR);
                            for bi in yi..yiend{
                                for bk in yk..ykend{
                                    for bj in (yj as isize)..yjend{
                                        let ik = bk+s*bi;
                                        let kj = bj+r*bk;
                                        let ij = bj+r*bi;
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
                    }
                }


                //Copy C workspace back to original matrix
                for (x,y) in cblk.iter_mut().zip(cb.iter()){
                    *x=*y;
                }
            }
        }
    }
}
