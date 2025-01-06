This is to test the quantized groundingdino model see: [`saburq/groundingdeno_model_quant_int8`](https://huggingface.co/saburq/groundingdeno_model_quant_int8).

## Usage

```bash
yarn install
npx nodemon app.js
```

```bash
# client
python draw_boxes.py
open output_with_boxes.png
```

## Example API

```bash
http://localhost:3000?model_name=object-detection&text=tree&image_uri=https://content.satimagingcorp.com/static/galleryimages/Satellite-Image-Paris-Pont-des-Arts-bridge.jpg
```