import http from 'http';
import querystring from 'querystring';
import url from 'url';

import { pipeline, env, SamModel, AutoProcessor, OwlViTImageProcessor, SamPreTrainedModel } from '@huggingface/transformers';

export class MyClassificationPipeline {
  static task = 'text-classification';
  static model = 'Xenova/distilbert-base-uncased-finetuned-sst-2-english';
  static instance = null;

  static async getInstance(progress_callback = null) {
    if (this.instance === null) {
      // NOTE: Uncomment this to change the cache directory
      // env.cacheDir = './.cache';

      this.instance = pipeline(this.task, this.model, { progress_callback });
    }

    return this.instance;
  }
}

export class GroundingDinoSingleton {
  static model_id = 'onnx-community/grounding-dino-tiny-ONNX';
  static model;
  static quantized = true;
  static task = 'zero-shot-object-detection';
  
  static async getInstance() {
    if (!this.model) {
      this.model = pipeline(this.task, this.model_id);
    }
    return this.model;
  }

  async run_inference(image_uri, candidate_labels) {
    const features = await this.model(image_uri, candidate_labels, {threshold: 0.3});
    console.log(features);
    return features;
  }
}

export class GroundingDinoSingletonWithOwlViT {
  static model_id = 'saburq/groundingdeno_model_quant_int8';
  static model;
  static quantized = true;
  static processor;

  static getInstance() {
    if (!this.model) {
      this.model = SamPreTrainedModel.from_pretrained(this.model_id, { quantized: this.quantized });
    }
    if (!this.processor) {
      this.processor = OwlViTImageProcessor.from_pretrained(this.model_id);
    }
    return Promise.all([this.model, this.processor]);
  }

  async run_inference(image_uri, text_prompt) {
    // process the image
    const image = await this.processor(text_prompt, image_uri, return_tensors="pt");
    const features = await this.model(image, text_prompt);
    console.log(features);
    return features;
  }
}

// add class for slimsam
export class SegmentAnythingSingleton {
  static model_id = 'Xenova/slimsam-77-uniform';
  static model;
  static processor;
  static quantized = true;

  static getInstance() {
    if (!this.model) {
      this.model = SamModel.from_pretrained(this.model_id, {
        quantized: this.quantized,
      });
    }
    if (!this.processor) {
      this.processor = AutoProcessor.from_pretrained(this.model_id);
    }

    return Promise.all([this.model, this.processor]);
  }
}