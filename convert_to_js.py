from tensorflowjs.converters import convert_tf_saved_model

# Convert the model
convert_tf_saved_model(
    saved_model_dir="my_plant_model",
    output_dir="my_plant_model_tfjs"
)
