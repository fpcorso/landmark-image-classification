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
    {{ landmark }}
  </section>
</template>

<script lang="ts">
import axios from 'axios';

export default {
  name: 'HomeView',
  data() {
    return {
      file: null,
      image: '',
      preview: false,
      landmark: ''
    };
  },
  methods: {
    readImage(event: Event) {
      this.file = event.target.files[0]; // Get the selected file
      if (this.file) {
        const reader = new FileReader(); // Create a new FileReader object
        reader.onload = () => {
          this.image = reader.result; // Assign the data URL to the image property
          this.preview = true;
        };
        reader.readAsDataURL(this.file); // Read the image file as a data URL
        this.makePrediction();
      }
    },
    makePrediction() {
      // Send image info to server using axios
      axios.post('http://localhost:8000/predict', {
        image: this.file
      }, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }).then((response) => {
        this.landmark = response.data.landmark;
      }).catch((error) => {
        console.log(error);
      });
      
    }
  }
};
</script>
