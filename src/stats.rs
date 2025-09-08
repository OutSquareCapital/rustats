use ndarray::NdFloat;

#[inline(always)]
pub fn var<T: NdFloat>(sum_simple: T, sum_square: T, obs: T) -> T {
    sum_square
        .sub(sum_simple.powi(2).div(obs))
        .div(obs.sub(T::one()))
}

#[inline(always)]
pub fn stdev<T: NdFloat>(sum_simple: T, sum_square: T, obs: T) -> T {
    var(sum_simple, sum_square, obs).sqrt()
}

#[inline(always)]
pub fn skew<T: NdFloat>(sum_simple: T, sum_square: T, sum_cube: T, obs: T) -> T {
    let mean = sum_simple.div(obs);
    let variance = var(sum_simple, sum_square, obs);
    let one = T::one();
    let two = one.add(one);
    let three = one.add(two);
    (obs.mul(obs.sub(one)))
        .sqrt()
        .mul(
            sum_cube
                .div(obs)
                .sub(mean.powi(3))
                .sub(three)
                .mul(mean)
                .mul(variance),
        )
        .div((obs.sub(two)).mul(variance.sqrt().powi(3)))
}

#[inline(always)]
pub fn kurtosis<T: NdFloat>(sum_simple: T, sum_square: T, sum_cube: T, sum_quad: T, obs: T) -> T {
    let mean = sum_simple.div(obs);
    let one = T::one();
    let two = one.add(one);
    let three = one.add(two);
    let four = two.add(two);
    let six = three.add(three);

    obs.sub(one)
        .mul(
            obs.add(one)
                .mul(
                    sum_quad
                        .div(obs)
                        .sub(four.mul(mean).mul(sum_cube.div(obs)))
                        .add(six.mul(mean.powi(2)).mul(sum_square.div(obs)))
                        .sub(three.mul(mean.powi(4))),
                )
                .div(var(sum_simple, sum_square, obs).powi(2))
                .sub(three.mul(obs.sub(one))),
        )
        .div(obs.sub(two).mul(obs.sub(three)))
}

#[inline(always)]
pub fn rank<T: NdFloat>(greater_count: T, equal_count: T, obs: T) -> T {
    let one = T::one();
    let two = one.add(one);
    let half = one.div(two);
    two.mul(
        (half.mul(greater_count.add(equal_count).sub(one)))
            .div(obs.sub(one))
            .sub(half),
    )
}
