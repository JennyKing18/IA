// Permite incluir el modelo .onnx como recurso de la app (assets/model/modelo_final.onnx)
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push('onnx');

module.exports = config;
