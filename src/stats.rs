#[inline(always)]
pub fn get_mean(mean_sum: f64, observation_count: usize) -> f64 {
    mean_sum / (observation_count as f64)
}
#[inline(always)]
pub fn get_var(mean_sum: f64, mean_square_sum: f64, observation_count: usize) -> f64 {
    let obs: f64 = observation_count as f64;
    mean_square_sum / obs - (mean_sum / obs).powi(2)
}

#[inline(always)]
pub fn get_std(mean_sum: f64, mean_square_sum: f64, observation_count: usize) -> f64 {
    get_var(mean_sum, mean_square_sum, observation_count).sqrt()
}

#[inline(always)]
pub fn get_skew(
    mean_sum: f64,
    mean_square_sum: f64,
    cubed_accumulator: f64,
    observation_count: usize
) -> f64 {
    let obs: f64 = observation_count as f64;
    let mean_value: f64 = get_mean(mean_sum, observation_count);
    let variance_value: f64 = get_var(mean_sum, mean_square_sum, observation_count);
    let skew_numerator: f64 =
        cubed_accumulator / obs - mean_value.powi(3) - 3.0 * mean_value * variance_value;

    let std_dev: f64 = variance_value.sqrt();

    let skew: f64 = ((obs * (obs - 1.0)).sqrt() * skew_numerator) / ((obs - 2.0) * std_dev.powi(3));
    skew
}

#[inline(always)]
pub fn get_kurtosis(
    mean_sum: f64,
    mean_square_sum: f64,
    cubed_accumulator: f64,
    quartic_accumulator: f64,
    observation_count: usize
) -> f64 {
    let obs: f64 = observation_count as f64;
    let mean_value: f64 = get_mean(mean_sum, observation_count);
    let variance_value: f64 = get_var(mean_sum, mean_square_sum, observation_count);
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
