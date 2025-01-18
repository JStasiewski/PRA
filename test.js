const tf = require('@tensorflow/tfjs'); // TensorFlow.js
const fs = require('fs'); // File system to read images
const { createCanvas, loadImage } = require('canvas'); // For image preprocessing

// Load the model
async function loadModel() {
    const model = await tf.loadGraphModel('file://./my_plant_model_tfjs/model.json');
    return model;
}

// Preprocess the image
async function preprocessImage(imagePath, targetSize = [224, 224]) {
    const img = await loadImage(imagePath); // Load image
    const canvas = createCanvas(targetSize[0], targetSize[1]);
    const ctx = canvas.getContext('2d');

    // Resize the image
    ctx.drawImage(img, 0, 0, targetSize[0], targetSize[1]);

    // Get pixel data and normalize
    const imageData = ctx.getImageData(0, 0, targetSize[0], targetSize[1]);
    let data = tf.browser.fromPixels(imageData).toFloat().div(255.0); // Scale to [0, 1]
    data = data.expandDims(0); // Add batch dimension: [1, height, width, channels]

    return data;
}

// Predict using the model
async function predictImage(imagePath) {
    const model = await loadModel();
    const input = await preprocessImage(imagePath);
    const predictions = model.predict(input);

    // Get top prediction
    const predictionArray = await predictions.array();
    const predictedIndex = tf.argMax(predictions, 1).arraySync()[0];
    const confidence = predictionArray[0][predictedIndex];

    // Class names (update as per your classes)
    const classNames = [
        "aloevera", "banan", "bilimbi", "cantaloupe", "cassava", "coconut", "corn", "cucumber",
        "curcuma", "eggplant", "galangal", "ginger", "guava", "kale",
        "longbeans", "mango", "melon", "orange", "paddy", "papaya",
        "peper chili", "pineapple", "pomelo", "shallot", "soybeans",
        "spinach", "sweet potatoes", "tobacco", "waterapple", "watermelon"
    ];

    console.log(`Predicted class: ${classNames[predictedIndex]} (Confidence: ${confidence.toFixed(2)})`);
}

// Run prediction on a sample image
const sampleImagePath = './image.png'; // Replace with your image path
predictImage(sampleImagePath);
