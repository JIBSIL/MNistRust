use bincode::*;
use itertools::izip;
use mnist::*;
use ndarray::prelude::*;
use rand::*;
use rand_distr::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::{fs::File, time::Instant};

const E: f64 = 2.718281828459045235;

// dataprep

fn mnist_decode() -> (Array3<f32>, Array2<f32>, Array3<f32>, Array2<f32>) {
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    // let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);

    let test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    return (train_data, train_labels, test_data, test_labels);
}

struct Neuron {
    weights: Vec<f64>, // One weight per input
    bias: f64,         // A single bias
}

impl Neuron {
    fn compute(&self, inputs: &[f64]) -> f64 {
        assert_eq!(
            self.weights.len(),
            inputs.len(),
            "Input size must match weights"
        );
        let weighted_sum: f64 = self.weights.iter().zip(inputs).map(|(w, &i)| w * i).sum();
        weighted_sum + self.bias
    }

    fn new(num_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weights = {
            let limit = (1.0 / num_inputs as f64).sqrt();
            (0..num_inputs)
                .map(|_| rng.gen_range(-limit..limit))
                .collect()
        };

        // avoid dead neurons in relu
        let bias = 0.01;

        Neuron { weights, bias }
    }
}

struct Dense {
    neurons: Vec<Neuron>,
}

#[derive(Deserialize, Serialize)]
struct DenseForBackprop {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    m_weights: Vec<Vec<f64>>,
    v_weights: Vec<Vec<f64>>,
    m_biases: Vec<f64>,
    v_biases: Vec<f64>,
    t: usize,
}

impl Dense {
    fn forward(&self, inputs: &[f64], activation: fn(f64) -> f64) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| activation(neuron.compute(inputs)))
            .collect()
    }
}

// this is super weird and experimental
// had to break out mr gbt for this one
// and im not sure if everything is correct
// so this is going into its own impl
impl DenseForBackprop {
    fn initialize(num_inputs: usize, num_neurons: usize) -> Self {
        let mut rng = rand::thread_rng();

        let stddev = (2.0 / num_inputs as f64).sqrt();
        let weights: Vec<Vec<f64>> = (0..num_neurons)
            .map(|_| {
                (0..num_inputs)
                    .map(|_| rng.sample(rand_distr::Normal::new(0.0, stddev).unwrap()))
                    .collect()
            })
            .collect();

        let biases: Vec<f64> = (0..num_neurons).map(|_| 0.01).collect();

        // Initialize m and v for Adam optimizer
        let m_weights = vec![vec![0.0; num_inputs]; num_neurons];
        let v_weights = vec![vec![0.0; num_inputs]; num_neurons];
        let m_biases = vec![0.0; num_neurons];
        let v_biases = vec![0.0; num_neurons];

        DenseForBackprop {
            weights,
            biases,
            m_weights,
            v_weights,
            m_biases,
            v_biases,
            t: 0,
        }
    }

    fn forward(&self, inputs: &[f64], activation: fn(f64) -> f64) -> Vec<f64> {
        let z: Vec<f64> = self
            .weights
            .par_iter()
            .map(|row| row.par_iter().zip(inputs).map(|(w, i)| w * i).sum::<f64>())
            .zip(&self.biases)
            .map(|(z, b)| z + b)
            .collect();

        z.par_iter().map(|&z| activation(z)).collect()
    }

    fn backward(
        &mut self,
        inputs: &[f64],
        delta_next: &[f64], // Error signal from the next layer
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let d_weights: Vec<Vec<f64>> = delta_next
            .par_iter()
            .map(|&delta| inputs.par_iter().map(|&input| delta * input).collect())
            .collect();

        let d_biases: Vec<f64> = delta_next.to_vec();

        let delta: Vec<f64> = self
            .weights
            .par_iter()
            .map(|weights| {
                weights
                    .par_iter()
                    .zip(delta_next)
                    .map(|(w, d_next)| w * d_next)
                    .sum()
            })
            .collect();

        (d_weights, d_biases, delta)
    }
}

#[derive(Deserialize, Serialize)]
struct Flatten;

impl Flatten {
    fn forward(&self, input: Vec<Vec<f32>>) -> Vec<f32> {
        input.into_iter().flatten().collect()
    }
}

// full net is just a bunch of layers
// todo:
#[derive(Deserialize, Serialize)]
struct NeuralNetwork {
    flatten: Flatten,
    dense_layers: Vec<DenseForBackprop>,
}

impl NeuralNetwork {
    fn total_params(&self) -> usize {
        self.dense_layers.iter().fold(0, |sum, layer| {
            sum + layer.weights.iter().flatten().count() + layer.biases.len()
        })
    }

    fn forward(&self, input: Vec<f32>, activation: fn(f64) -> f64) -> Vec<f64> {
        let mut current_output: Vec<f64> = input.iter().map(|&x| f64::from(x)).collect();
        // println!("input size2 {:?}", current_output.iter().count());
        for layer in &self.dense_layers {
            current_output = layer.forward(&current_output, activation);
        }
        current_output
    }

    fn backward(
        &mut self,
        inputs: Vec<f32>,                 // Original input to the network
        label: usize,                     // True label
        activation_prime: fn(f64) -> f64, // Derivative of activation function
        optimizer: &AdamOptimizer,        // Optimizer for updating weights
    ) -> f64 {
        let mut activations = vec![];
        let mut zs = vec![];
        let mut current_input: Vec<f64> = inputs.iter().map(|&x| f64::from(x)).collect();

        for layer in &self.dense_layers {
            let z: Vec<f64> = layer
                .weights
                .par_iter()
                .map(|row| {
                    row.iter()
                        .zip(&current_input)
                        .map(|(w, i)| w * i)
                        .sum::<f64>()
                })
                .zip(&layer.biases)
                .map(|(z, b)| z + b)
                .collect();

            let a: Vec<f64> = z.iter().map(|&z| leaky_relu(z)).collect();

            zs.push(z);
            activations.push(current_input.clone());
            current_input = a;
        }

        let logits = current_input.clone();
        let loss = sparse_categorical_crossentropy(&logits, label);

        let mut delta_next: Vec<f64> = logits
            .iter()
            .enumerate()
            .map(|(i, &y_hat)| if i == label { y_hat - 1.0 } else { y_hat })
            .collect();

        for (i, layer) in self.dense_layers.iter_mut().enumerate().rev() {
            layer.t += 1; // Increment time step for Adam optimizer
            let (d_weights, d_biases, delta) = layer.backward(&activations[i], &delta_next);
            // Update weights and biases using Adam optimizer
            optimizer.update_vec_vec(
                &mut layer.weights,
                &d_weights,
                &mut layer.m_weights,
                &mut layer.v_weights,
                layer.t,
            );
            optimizer.update_vec(
                &mut layer.biases,
                &d_biases,
                &mut layer.m_biases,
                &mut layer.v_biases,
                layer.t,
            );
            delta_next = delta; // Pass error signal to the previous layer
        }

        loss
    }

    fn train_step(
        self: &mut NeuralNetwork,
        info_in: Vec<(Vec<Vec<f32>>, f32)>,
        optimizer: &AdamOptimizer,
    ) -> f64 {
        let mut total_loss = 0.0;

        let input_loss_label: Vec<(Vec<f32>, f64, f32)> = info_in
            .par_iter()
            .map(|(input, label)| {
                let flattened_input = self.flatten.forward(input.clone());
                let logits = self.forward(flattened_input.clone(), leaky_relu);
                let loss = sparse_categorical_crossentropy(&logits, *label as usize);
                (flattened_input, loss, *label)
            })
            .collect();

        for (inputs, loss, label) in input_loss_label {
            self.backward(inputs, label as usize, leaky_relu_prime, optimizer);
            total_loss += loss;
        }

        total_loss / info_in.len() as f64
    }
}

// relu activation thx mr gbt
// fn relu(x: f64) -> f64 {
//     x.max(0.0)
// }

// fn relu_prime(x: f64) -> f64 {
//     if x > 0.0 {
//         1.0
//     } else {
//         0.0
//     }
// }

// just experiment with diff leaky relu params
fn leaky_relu(x: f64) -> f64 {
    let alpha = 0.01;
    if x > 0.0 {
        x
    } else {
        alpha * x
    }
}

fn leaky_relu_prime(x: f64) -> f64 {
    let alpha = 0.01;
    if x > 0.0 {
        1.0
    } else {
        alpha
    }
}

fn sigma_function(x: f64) -> f64 {
    let res = 1_f64 / (1_f64 + f64::powf(E, 0_f64 - x));
    if res.is_nan() {
        0_f64
    } else {
        res
    }
}

fn sigma_function_prime(x: f64) -> f64 {
    let epownegx = f64::powf(E, 0_f64 - x);
    let epow2 = 1_f64 + epownegx;
    let res = epownegx / (epow2 * epow2);
    if res.is_nan() {
        0_f64
    } else {
        res
    }
}

fn ndarray_process<T: Clone>(array: Array2<T>) -> Vec<Vec<T>> {
    array.axis_iter(Axis(0)).map(|row| row.to_vec()).collect()
}

fn sparse_categorical_crossentropy(logits: &[f64], true_label: usize) -> f64 {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_logits: Vec<f64> = logits.iter().map(|&z| (z - max_logit).exp()).collect();
    let sum_exp_logits: f64 = exp_logits.iter().sum();
    let softmax_true = exp_logits[true_label] / sum_exp_logits;

    -softmax_true.ln() // Negative log of the true class probability
}

fn sparse_categorical_accuracy(logits: &[f64], true_label: usize) -> f64 {
    let predicted_label = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap();

    if predicted_label == true_label {
        1.0
    } else {
        0.0
    }
}

// adamgpt
struct AdamOptimizer {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}

impl AdamOptimizer {
    fn new(learning_rate: f64) -> Self {
        AdamOptimizer {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-7,
        }
    }

    fn update_vec_vec(
        &self,
        params: &mut Vec<Vec<f64>>,
        grads: &Vec<Vec<f64>>,
        m: &mut Vec<Vec<f64>>,
        v: &mut Vec<Vec<f64>>,
        t: usize,
    ) {
        for (param_row, grad_row, m_row, v_row) in
            izip!(params.iter_mut(), grads.iter(), m.iter_mut(), v.iter_mut())
        {
            for (param, grad, m_elem, v_elem) in izip!(
                param_row.iter_mut(),
                grad_row.iter(),
                m_row.iter_mut(),
                v_row.iter_mut()
            ) {
                *m_elem = self.beta1 * *m_elem + (1.0 - self.beta1) * *grad;
                *v_elem = self.beta2 * *v_elem + (1.0 - self.beta2) * grad.powi(2);

                let m_hat = *m_elem / (1.0 - self.beta1.powi(t as i32));
                let v_hat = *v_elem / (1.0 - self.beta2.powi(t as i32));

                *param -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
    }

    fn update_vec(
        &self,
        params: &mut Vec<f64>,
        grads: &Vec<f64>,
        m: &mut Vec<f64>,
        v: &mut Vec<f64>,
        t: usize,
    ) {
        for (param, grad, m_elem, v_elem) in
            izip!(params.iter_mut(), grads.iter(), m.iter_mut(), v.iter_mut())
        {
            *m_elem = self.beta1 * *m_elem + (1.0 - self.beta1) * *grad;
            *v_elem = self.beta2 * *v_elem + (1.0 - self.beta2) * grad.powi(2);

            let m_hat = *m_elem / (1.0 - self.beta1.powi(t as i32));
            let v_hat = *v_elem / (1.0 - self.beta2.powi(t as i32));

            *param -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

fn main() {
    // TF-KERAS code:
    // model = tf.keras.models.Sequential([
    //     tf.keras.layers.Flatten(input_shape=(28, 28)),
    //     tf.keras.layers.Dense(128, activation='relu'),
    //     tf.keras.layers.Dense(10)
    //   ])
    //   model.compile(
    //       optimizer=tf.keras.optimizers.Adam(0.001),
    //       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    //       metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    //   )

    //   model.fit(
    //       ds_train,
    //       epochs=6,
    //       validation_data=ds_test,
    //   )
    let (train_data, train_labels, test_data, test_labels) = mnist_decode();
    // let train_data = train_data.map(|x| *x as f32 / 255.0);
    let seq_1 = Flatten {};
    let seq_2 = DenseForBackprop::initialize(784, 128);
    let seq_3 = DenseForBackprop::initialize(128, 10);

    let mut model = NeuralNetwork {
        flatten: seq_1,
        dense_layers: vec![seq_2, seq_3],
    };

    let optimizer = AdamOptimizer::new(0.001);

    let batch_size = 64;
    let training_vec = train_data
        .outer_iter()
        .map(|x| ndarray_process(x.to_owned()))
        .collect::<Vec<Vec<Vec<f32>>>>();
    // let training_chunks = training_vec.chunks(batch_size);

    let training_label_vec: Vec<f32> = train_labels
        .outer_iter()
        .map(|x| *x.get(0).unwrap())
        .collect();

    let final_training_data: Vec<Vec<(Vec<Vec<f32>>, f32)>> = training_vec
        .iter()
        .zip(training_label_vec)
        .map(|(x, y)| (x.clone(), y))
        .collect::<Vec<(Vec<Vec<f32>>, f32)>>()
        .chunks(batch_size)
        .map(|x| x.to_vec())
        .collect();

    let max_iter_count = final_training_data.clone().into_iter().count();

    for ii in 1..6 {
        for (i, input) in final_training_data.clone().into_iter().enumerate() {
            // println!("input size {:?}", input.iter().count());
            let now = Instant::now();
            let loss = model.train_step(input, &optimizer);
            let elapsed = now.elapsed();
            if i % 50 == 0 {
                println!(
                    "Epoch: {:?}. Step: {:?}/{:?}. Loss: {:?}. Time: {:?}, Total: {:?}",
                    ii,
                    i,
                    max_iter_count,
                    loss,
                    elapsed,
                    elapsed * 50
                );
            }
        }
    }

    println!("Training successful; saving to weights.bin");

    if Path::new("weights.bin").exists() {
        println!("Removing old weights.bin...");
        fs::remove_file("weights.bin").unwrap();
    }

    let serialized_model = bincode::serialize(&model).unwrap();
    let mut outfile = File::create("weights.bin").unwrap();
    outfile.write(serialized_model.as_slice()).unwrap();
    println!("Wrote out weights.bin!")
}
