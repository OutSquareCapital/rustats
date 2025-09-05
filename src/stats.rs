use std::ops::{Add, Div, Mul, Sub};

#[inline(always)]
pub fn var(sum_simple: f64, sum_square: f64, obs: f64) -> f64 {
    sum_square
        .sub(sum_simple.powi(2).div(obs))
        .div(obs.sub(1.0))
}

#[inline(always)]
pub fn stdev(sum_simple: f64, sum_square: f64, obs: f64) -> f64 {
    var(sum_simple, sum_square, obs).sqrt()
}

#[inline(always)]
pub fn skew(sum_simple: f64, sum_square: f64, sum_cube: f64, obs: f64) -> f64 {
    (obs.mul(obs.sub(1.0)))
        .sqrt()
        .mul(
            sum_cube
                .div(obs)
                .sub(sum_simple.div(obs).powi(3))
                .sub(3.0)
                .mul(sum_simple.div(obs))
                .mul(var(sum_simple, sum_square, obs)),
        )
        .div((obs.sub(2.0)).mul(var(sum_simple, sum_square, obs).sqrt().powi(3)))
}

#[inline(always)]
pub fn kurtosis(sum_simple: f64, sum_square: f64, sum_cube: f64, sum_quad: f64, obs: f64) -> f64 {
    obs.sub(1.0)
        .mul(
            obs.add(1.0)
                .mul(
                    sum_quad
                        .div(obs)
                        .sub(4.0.mul(sum_simple.div(obs)).mul(sum_cube.div(obs)))
                        .add(
                            6.0.mul(sum_simple.div(obs).powi(2))
                                .mul(sum_square.div(obs)),
                        )
                        .sub(3.0.mul(sum_simple.div(obs).powi(4))),
                )
                .div(var(sum_simple, sum_square, obs).powi(2))
                .sub(3.0.mul(obs.sub(1.0))),
        )
        .div(obs.sub(2.0).mul(obs.sub(3.0)))
}

#[inline(always)]
pub fn rank(greater_count: usize, equal_count: usize, obs: f64) -> f64 {
    2.0.mul(
        (0.5.mul((greater_count.add(equal_count).sub(1)) as f64))
            .div(obs.sub(1.0))
            .sub(0.5),
    )
}
