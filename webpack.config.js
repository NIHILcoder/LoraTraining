const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = (env, argv) => {
  const isDev = argv.mode === 'development';

  const webConfig = {
    name: 'web',
    target: 'electron-renderer',
    entry: { renderer: './src/index.tsx' },
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: isDev ? '[name].js' : '[name].[contenthash].js',
      clean: true,
      publicPath: '/',
    },
    resolve: {
      extensions: ['.tsx', '.ts', '.js', '.jsx'],
      alias: {
        '@': path.resolve(__dirname, 'src'),
      },
    },
    module: {
      rules: [
        {
          test: /\.(ts|tsx)$/,
          exclude: /node_modules/,
          use: 'babel-loader',
        },
        {
          test: /\.css$/,
          use: [
            isDev ? 'style-loader' : MiniCssExtractPlugin.loader,
            'css-loader',
          ],
        },
        {
          test: /\.(png|jpg|jpeg|webp|gif|svg)$/i,
          type: 'asset/resource',
        },
      ],
    },
    plugins: [
      new HtmlWebpackPlugin({
        template: './public/index.html',
        title: 'LoRA Training Dashboard',
      }),
      ...(!isDev
        ? [
            new MiniCssExtractPlugin({
              filename: '[name].[contenthash].css',
            }),
          ]
        : []),
    ],
    devServer: {
      port: 3005,
      hot: true,
      historyApiFallback: true,
      open: false,
      devMiddleware: {
        writeToDisk: true,
      },
    },
    devtool: isDev ? 'eval-source-map' : 'source-map',
  };

  const electronConfig = {
    name: 'electron',
    target: 'electron-main',
    entry: './src/main.ts',
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: 'main.js',
      clean: false, // Prevent race conditions with web config
    },
    resolve: {
      extensions: ['.ts', '.js'],
    },
    module: {
      rules: [
        {
          test: /\.ts$/,
          exclude: /node_modules/,
          use: 'babel-loader',
        },
      ],
    },
    devtool: isDev ? 'eval-source-map' : 'source-map',
    externals: {
      fsevents: "require('fsevents')",
    },
  };

  if (process.env.WEBPACK_SERVE) {
    webConfig.output.clean = false; // Prevent it from deleting main.js
    return webConfig;
  }

  return [webConfig, electronConfig];
};
