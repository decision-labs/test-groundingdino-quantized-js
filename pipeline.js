import http from 'http';
import querystring from 'querystring';
import url from 'url';

import { pipeline, env, SamModel, AutoProcessor } from '@huggingface/transformers';

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
  static model_id = 'saburq/groundingdeno_model_quant_int8';
  static model;
  static processor;
  static quantized = true;
  static task = 'object-detection';

  static async getInstance() {
    if (!this.model) {
      this.model = pipeline(this.task, this.model_id, { quantized: this.quantized });
    }
    return this.model;
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