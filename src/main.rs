use std::env;
use std::time::{Instant};

pub mod reference;
pub mod gemm_iterator;

//use reference::gemm;
use gemm_iterator::gemm;

fn main() {
    let args: Vec<String> = env::args().collect();
    let m=args[1].parse::<usize>().unwrap();
    let n=m as usize;
    let k=m;
    let lda=k;
    let ldb=n;
    let ldc=n;
    let nruns=100;
    let mut c1 = vec![0.0;m*n];
    let mut c2 = vec![0.0;m*n];
    let a : Vec<f64> = (0..m*n).map(|x|{ (x as f64).sin()+2.0}).collect();
    let b : Vec<f64> = (0..m*n).map(|x|{ (x as f64).cos()+2.0}).collect();

    /*First do a quick run against reference implementation
     * to see they do the same thing.*/
    reference::gemm(m,n,k,&a,lda,&b,ldb,&mut c1,ldc);
    gemm(m,n,k,&a,lda,&b,ldb,&mut c2,ldc);


    let mut max_relerr = 0.0;
    for (&x,&y) in c1.iter().zip(c2.iter()){
        let err = (x-y).abs();
        let minxy = if x.abs()<y.abs() { x.abs() } else { y.abs() };
        let relerr = err/minxy;
        max_relerr = if max_relerr<relerr { relerr } else { max_relerr };
    }
    println!("Maximum relative error    =     {}",max_relerr);




    //Now time the actual implementation
    let times = {
        let mut times = Vec::<f64>::new();
        for _ in 0..nruns{
            let now = Instant::now();
            gemm(m,n,k,&a,lda,&b,ldb,&mut c2,ldc);
            times.push(now.elapsed().as_secs_f64());
        }
        times.sort_by(|x,y|x.partial_cmp(y).unwrap());
        times
    };
    {
        let min=times[0];
        let max=*times.last().unwrap();
        let avg=times.iter().fold(0.0,|acc,x|acc+x)/(times.len() as f64);
        let std=(times.iter().map(|x|(avg-x)*(avg-x)).fold(0.0,|acc,x|acc+x)/(times.len() as f64)).sqrt();
        print!("Reference:    min = {},   max = {},  avg = {},  std = {}\n",min,max,avg,std);
    }



}
