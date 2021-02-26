use ndarray::{ArrayViewMut2,ArrayView2,Zip,s};


pub fn gemm(m : usize, n : usize, p : usize, a : &[f64], _lda : usize, b : &[f64], _ldb : usize, c : &mut [f64], _ldc : usize) -> () {

    //Wrap data slices into ArrayViews from ndarray crate.
    //This gives good iterators over subarrays
    let ablk_big = ArrayView2::<f64>::from_shape((m,p),a).unwrap();
    let bblk_big = ArrayView2::<f64>::from_shape((p,m),b).unwrap();
    let mut cblk_big = ArrayViewMut2::<f64>::from_shape((m,n),c).unwrap();


    const BI : usize = 64;
    const BJ : usize = 64;
    const BK : usize = 64;

    for i in (0..m).step_by(BI){
        for j in (0..n).step_by(BJ){
            for k in (0..p).step_by(BK){
                let iend=std::cmp::min(m,i+BI);
                let jend=std::cmp::min(n,j+BJ);
                let kend=std::cmp::min(p,k+BK);
                let ablk = ablk_big.slice(s![i..iend,k..kend]);
                let bblk = bblk_big.slice(s![k..kend,j..jend]);
                let mut cblk = cblk_big.slice_mut(s![i..iend,j..jend]);

                for (mut ci,ai) in cblk.genrows_mut().into_iter().zip(ablk.genrows().into_iter()){
                    for (aik,bk) in ai.iter().zip(bblk.genrows().into_iter()){
                        Zip::from(&mut ci).and(&bk).apply(|x,y|{*x+=(*y)*(*aik);});
                    }
                }
            }
        }
    }
}
