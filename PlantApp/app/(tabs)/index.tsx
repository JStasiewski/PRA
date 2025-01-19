import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Button,
  StyleSheet,
  Text,
  TouchableOpacity,
  View
} from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { useRouter } from 'expo-router'; // Import useRouter

// TensorFlow.js for React Native:
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native';

const modelJson = require('../../assets/my_plant_model_tfjs/model.json');
const modelWeights1 = require('../../assets/my_plant_model_tfjs/group1-shard1of3.bin');
const modelWeights2 = require('../../assets/my_plant_model_tfjs/group1-shard2of3.bin');
const modelWeights3 = require('../../assets/my_plant_model_tfjs/group1-shard3of3.bin');

// Example class labels
const classNames = [
  'aloe', 'banan', 'averrhoa', 'cantaloupe', 'cassava', 'coconut', 'corn',
  'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger', 'guava', 'kale',
  'asparagus', 'mango', 'melon', 'orange', 'rice', 'papaya',
  'pepper', 'pineapple', 'pomelo', 'shallot', 'soy bean',
  'spinach', 'sweet potato', 'tobacco', 'syzygium', 'watermelon'
];

export default function App() {
  const [facing, setFacing] = useState<CameraType>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<any>(null); // reference to CameraView
  const [model, setModel] = useState<tf.GraphModel | null>(null);

  // Track loading state and prediction result
  const [loading, setLoading] = useState(false);
  const [predictionText, setPredictionText] = useState<string>('');
  const router = useRouter();

  // Load TensorFlow model once on startup
  useEffect(() => {
    (async () => {
      try {
        await tf.ready();

        const loadedModel = await tf.loadGraphModel(
          bundleResourceIO(modelJson, [modelWeights1, modelWeights2, modelWeights3])
        );

        setModel(loadedModel);
        console.log('Model loaded successfully!');
      } catch (err) {
        console.error('Failed to load model:', err);
      }
    })();
  }, []);

  // Handle camera permissions
  if (!permission) {
    // Camera permissions are still loading
    return <View />;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet
    return (
      <View style={styles.container}>
        <Text style={styles.message}>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="grant permission" />
      </View>
    );
  }

  // Flip between front & back camera
  function toggleCameraFacing() {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  }

  async function handleSnapAndPredict() {
    if (!cameraRef.current || !model) {
      console.log('Camera or model not ready.');
      return;
    }

    try {
      setLoading(true);
      setPredictionText('');

      // 1) Take photo
      const photo = await cameraRef.current.takePictureAsync({ quality: 1 });
      console.log('Photo URI:', photo.uri);

      // 2) Fetch arrayBuffer from the photo URI
      const response = await fetch(photo.uri);
      const arrayBuffer = await response.arrayBuffer();

      // 3) Decode the JPEG (React Native)
      const imageArray = new Uint8Array(arrayBuffer);
      let imgTensor = decodeJpeg(imageArray);

      // 4) Resize & normalize
      imgTensor = tf.image.resizeBilinear(imgTensor, [224, 224]);
      imgTensor = imgTensor.div(255).expandDims(0);

      // 5) Predict
      const predictions = model.predict(imgTensor) as tf.Tensor;
      const predictionArray = (await predictions.array()) as number[][];
      const predictedIndex = tf.argMax(predictions, 1).dataSync()[0];
      const confidence = predictionArray[0][predictedIndex];
      const className = classNames[predictedIndex];

      const confidencePercent = (confidence * 100).toFixed(2) + '%';
      console.log(`Predicted: ${className} (${confidencePercent})`);

      // 6) Save result to state
      setPredictionText(`${className} (${confidencePercent})`);
      router.push({pathname: '/(tabs)/explore', params: {predictedClass: className, confidence: confidencePercent}})

    } catch (err) {
      console.error('Error taking photo or predicting:', err);
      setPredictionText('Prediction error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing={facing}
      >
        <View style={styles.overlayContainer}>
          {loading ? (
            <ActivityIndicator size="large" color="#fff" />
          ) : predictionText ? (
            <Text style={styles.predictionText}>{predictionText}</Text>
          ) : null}
        </View>

        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={toggleCameraFacing}>
            <Text style={styles.text}>Flip Camera</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={handleSnapAndPredict}>
            <Text style={styles.text}>Snap & Predict</Text>
          </TouchableOpacity>
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
  },
  message: {
    textAlign: 'center',
    paddingBottom: 10,
  },
  camera: {
    flex: 1,
  },
  overlayContainer: {
    position: 'absolute',
    top: 40,
    width: '100%',
    alignItems: 'center',
    zIndex: 10,
  },
  predictionText: {
    color: 'white',
    fontSize: 20,
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    overflow: 'hidden',
  },
  buttonContainer: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 64,
    alignItems: 'flex-end',
  },
  button: {
    flex: 1,
    alignItems: 'center',
  },
  text: {
    fontSize: 20,
    color: 'white',
    fontWeight: 'bold',
  },
});
