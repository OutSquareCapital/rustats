#[inline(always)]
pub fn mean(mean_sum: f64, observation_count: usize) -> f64 {
    mean_sum / (observation_count as f64)
}
#[inline(always)]
pub fn var(mean_sum: f64, mean_square_sum: f64, observation_count: usize) -> f64 {
    let obs: f64 = observation_count as f64;
    (mean_square_sum / obs - (mean_sum / obs).powi(2)) * (obs / (obs - 1.0))
}

#[inline(always)]
pub fn stdev(mean_sum: f64, mean_square_sum: f64, observation_count: usize) -> f64 {
    var(mean_sum, mean_square_sum, observation_count).sqrt()
}

#[inline(always)]
pub fn skew(
    mean_sum: f64,
    mean_square_sum: f64,
    cubed_accumulator: f64,
    observation_count: usize
) -> f64 {
    let obs: f64 = observation_count as f64;
    let mean_value: f64 = mean(mean_sum, observation_count);
    let variance_value: f64 = var(mean_sum, mean_square_sum, observation_count);
    let skew_numerator: f64 =
        cubed_accumulator / obs - mean_value.powi(3) - 3.0 * mean_value * variance_value;

    let std_dev: f64 = variance_value.sqrt();

    let skew: f64 = ((obs * (obs - 1.0)).sqrt() * skew_numerator) / ((obs - 2.0) * std_dev.powi(3));
    skew
}

#[inline(always)]
pub fn kurtosis(
    mean_sum: f64,
    mean_square_sum: f64,
    cubed_accumulator: f64,
    quartic_accumulator: f64,
    observation_count: usize
) -> f64 {
    let obs: f64 = observation_count as f64;
    let mean_value: f64 = mean(mean_sum, observation_count);
    let variance_value: f64 = var(mean_sum, mean_square_sum, observation_count);
    let skew_numerator: f64 =
        cubed_accumulator / obs - mean_value.powi(3) - 3.0 * mean_value * variance_value;

    let kurtosis_term: f64 =
        quartic_accumulator / obs -
        mean_value.powi(4) -
        6.0 * variance_value * mean_value.powi(2) -
        4.0 * skew_numerator * mean_value;

    let kurt: f64 =
        (((obs * obs - 1.0) * kurtosis_term) / variance_value.powi(2) - 3.0 * (obs - 1.0).powi(2)) /
        ((obs - 2.0) * (obs - 3.0));
    kurt
}

#[inline(always)]
pub fn rank(greater_count: usize, equal_count: usize, valid_count: usize) -> f32 {
    if valid_count == 1 {
        0.0
    } else {
        let raw_rank: f32 = 0.5 * ((greater_count + equal_count - 1) as f32);
        let normalized_rank: f32 = 2.0 * (raw_rank / ((valid_count - 1) as f32) - 0.5);
        normalized_rank
    }
}
