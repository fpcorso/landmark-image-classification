<template>
  <section class="hero">
    <div class="hero-body">
      <p class="title is-2">
        What Landmark?
      </p>
      <p class="subtitle">
        Identify a landmark from a photo
      </p>
    </div>
  </section>
  <section class="section">
    <h2 class="title is-4">Upload Your Image</h2>
    <div class="file is-boxed">
      <label class="file-label">
        <input class="file-input" type="file" name="file" @change="readImage">
        <span class="file-cta">
          <span class="file-icon">
            <i class="fas fa-upload"></i>
          </span>
          <span class="file-label">
            Choose a file…
          </span>
        </span>
      </label>
    </div>
    <figure class="image is-128x128" v-if="preview">
      <img :src="image">
    </figure>
  </section>
</template>

<script lang="ts">
import axios from 'axios';

export default {
  name: 'HomeView',
  data() {
    return {
      image: '',
      preview: false
    };
  },
  methods: {
    readImage(event: Event) {
      const file = event.target.files[0]; // Get the selected file
      console.log(file);
      if (file) {
        const reader = new FileReader(); // Create a new FileReader object
        reader.onload = () => {
          this.image = reader.result; // Assign the data URL to the image property
          this.preview = true;
        };
        reader.readAsDataURL(file); // Read the image file as a data URL
      }
    },
    makePrediction() {
      // Send image info to server using axios
      axios.post('http://localhost:5000/predict', {
        image: this.image
      }).then((response) => {
        console.log(response);
      });
      
    }
  }
};
</script>
