use ndarray::{Array2, Array3, ArrayView2};
use rten::{NodeId, ValueOrView};
use rten_tensor::{prelude::*, *};

use neothesia_ai::{
    FRAME_THRESHOLD, ONSET_THRESHOLD, SEGMENT_SAMPLES, deframe, enframe,
    get_binarized_output_from_regression, note_detection_with_onset_offset_regress,
};

mod args;
mod audio;

fn main() -> anyhow::Result<()> {
    let args = args::Args::get_from_env()?;

    let input = audio::load(&args.input)?;

    let input = ArrayView2::from_shape([1, input.len()], &input)?;
    let input = enframe(&input, SEGMENT_SAMPLES);
    let input = input.as_slice().unwrap().to_vec();

    let input = Tensor::from_data(&[input.len() / SEGMENT_SAMPLES, SEGMENT_SAMPLES], input);

    let model = rten::Model::load_file(&args.model)?;

    let inputs: Vec<(NodeId, ValueOrView)> = vec![(model.input_ids()[0], input.view().into())];

    let [
        reg_onset_output,
        reg_offset_output,
        frame_output,
        _velocity_output,
        _reg_pedal_onset_output,
        _reg_pedal_offset_output,
        _pedal_frame_output,
    ] = model.run_n::<7>(inputs, model.output_ids().try_into()?, None)?;

    let (onset_output, onset_shift_output) = {
        let output = reg_onset_output.into_tensor::<f32>().unwrap();
        let shape: [usize; 3] = output.shape().try_into().unwrap();
        let reg_onset_output = Array3::from_shape_vec(shape, output.to_vec()).unwrap();
        let reg_onset_output: Array2<_> = deframe(&reg_onset_output);

        get_binarized_output_from_regression(&reg_onset_output.view(), ONSET_THRESHOLD, 2)
    };

    let (offset_output, offset_shift_output) = {
        let output = reg_offset_output.into_tensor::<f32>().unwrap();
        let shape: [usize; 3] = output.shape().try_into().unwrap();
        let reg_offset_output: Array3<_> = Array3::from_shape_vec(shape, output.to_vec()).unwrap();
        let reg_offset_output: Array2<_> = deframe(&reg_offset_output);

        let offset_threshold = 0.2;
        get_binarized_output_from_regression(&reg_offset_output.view(), offset_threshold, 4)
    };

    let frame_output: Array3<_> = {
        let output = frame_output.into_tensor::<f32>().unwrap();
        let shape: [usize; 3] = output.shape().try_into().unwrap();
        Array3::from_shape_vec(shape, output.to_vec()).unwrap()
    };
    let frame_output: Array2<_> = deframe(&frame_output);

    let file = note_detection_with_onset_offset_regress(
        frame_output.view(),
        onset_output.view(),
        onset_shift_output.view(),
        offset_output.view(),
        offset_shift_output.view(),
        (), // velocity_output,
        FRAME_THRESHOLD,
    );

    file.save(args.output)?;

    Ok(())
}
