// metro.config.js
const { getDefaultConfig } = require('expo/metro-config');

module.exports = (async () => {
    const config = await getDefaultConfig(__dirname);

    // Add .bin to assetExts
    config.resolver.assetExts.push('bin');

    return config;
})();
