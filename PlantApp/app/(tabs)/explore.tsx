import { useState, useEffect } from 'react';
import { useLocalSearchParams } from 'expo-router';
import { View, Text, StyleSheet, ScrollView, ActivityIndicator } from 'react-native';
import React from 'react';

// Perenual API endpoint and key (replace with your actual API key)
const API_URL = 'https://perenual.com/api/species-list';
const API_KEY = 'sk-2rTB678c5d8e5e65a8282'; // Replace with your actual API key

export default function ExploreScreen() {
  const { predictedClass } = useLocalSearchParams();
  const [plantData, setPlantData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    async function fetchPlantData() {
      try {
        console.log(`${API_URL}?key=${API_KEY}&q=${predictedClass}`);
        const response = await fetch(`${API_URL}?key=${API_KEY}&q=${predictedClass}`);
        const data = await response.json();
        console.log("API Response:", data);

        if (data && data.data && data.data.length > 0) {
          setPlantData(data.data[0]); // Use the first match
        } else {
          setPlantData(null); // No data found
        }
      } catch (err) {
        setError('Failed to fetch plant data.');
        console.error('API Error:', err);
      } finally {
        setLoading(false);
      }
    }

    if (predictedClass) {
      fetchPlantData();
    }
  }, [predictedClass]);

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#000" />
        <Text style={styles.text}>Loading plant data...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.errorContainer}>
        <Text style={styles.errorText}>{error}</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Predicted Class: {predictedClass}</Text>
      {plantData ? (
        <>
          <Text style={styles.subtitle}>Plant Details:</Text>
          <Text style={styles.text}>Common Name: {plantData.common_name}</Text>
          <Text style={styles.text}>Scientific Name: {plantData.scientific_name}</Text>
          <Text style={styles.text}>Watering: {plantData.watering}</Text>
          <Text style={styles.text}>Sunlight: {plantData.sunlight.join(', ')}</Text>
          <Text style={styles.text}>Cycle: {plantData.cycle}</Text>
        </>
      ) : (
        <Text style={styles.text}>No additional data found for this plant.</Text>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  subtitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 16,
    marginBottom: 8,
    color: "#fff",
  },
  text: {
    fontSize: 16,
    marginBottom: 8,
    color: "#fff",
  },
  errorText: {
    fontSize: 16,
    color: 'red',
  }
});
