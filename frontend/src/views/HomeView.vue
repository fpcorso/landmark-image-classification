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
    <div class="columns">

      <div class="column">
        <h2 class="title is-4">Upload Your Image</h2>
        <figure class="image mb-4" v-if="preview">
          <img :src="image" class="landmark-image">
        </figure>
        <div class="file is-boxed">
          <label class="file-label">
            <input class="file-input" type="file" name="file" @change="readImage">
            <span class="file-cta">
              <span class="file-icon">
                <i class="fas fa-upload"></i>
              </span>
              <span class="file-label">
                Choose an image...
              </span>
            </span>
          </label>
        </div>
      </div>

      <div class="column" v-if="landmarkKnown">
        <h2 class="title is-4">Landmark Info</h2>
        <p>This landmark is <span class="landmark-name">{{ landmark }}</span>.</p>
      </div>
    </div>
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
      landmark: '',
      landmarkKnown: false
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
        this.landmarkKnown = true;
      }).catch((error) => {
        console.log(error);
      });
      
    }
  }
};
</script>

<style scoped>
  .landmark-image {
    max-height: 200px;
    width: 100%;
  }

  label.file-label {
    width: 100%;
  }
  .landmark-name {
    font-weight: bold;
  }
</style>
