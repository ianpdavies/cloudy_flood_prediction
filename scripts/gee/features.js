/* Visualizes flood conditioning features over an example area */

var region = geometry
var palettes = require('users/gena/packages:palettes');

// Topo features
var elevation = ee.Image("USGS/NED").select('elevation')
// var elevation = ee.Image("USGS/SRTMGL1_003").select('elevation')

var slope = ee.Terrain.slope(elevation);
var curve  = (elevation.convolve(ee.Kernel.laplacian8())).resample().select(['elevation'],['curve']); //slope of the slope
curve = curve
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 1024
    })

var flowAcc = ee.ImageCollection("users/imerg/flow_acc_3s").mosaic() // 3 arcseconds
var flowAccClip = flowAcc.clip(region);
var imageValue1 = ee.Image.constant(1);
var flowAccClipPlusOne = flowAccClip.add(imageValue1); // so no 0s when taking the log
var pixelArea = ee.Image.pixelArea().mask(flowAccClip.mask());
var catchmentArea = flowAccClipPlusOne.multiply(pixelArea); //CA= (flow accum + 1 * cell^2)
var nonZeroSlopes = slope.add(ee.Image.constant(0.0000001)); //so low slopes don't get left out
var slopeRadians = nonZeroSlopes.multiply(3.141592653/280);
var tanSlope = slopeRadians.tan();

  /* http://www.gitta.info/TerrainAnalyi/en/html/HydroloAppls_learningObject4.xhtml
  topographic index is LN([flowaccum+1*cellarea]/tanslope)
  It is defined as ln(a/tanβ) where a is the local
  upslope area draining through a certain point per unit contour
  length and tanβ is the local slope.
  note that both TWI and SPI assume steady state */

var topoIndex = catchmentArea.divide(tanSlope);
var twi = topoIndex.log().select(['b1'],['twi']);
var streamPower = catchmentArea.multiply(tanSlope);
var spi = streamPower.clip(region).select(['b1'],['spi']);

var sinSlope = slopeRadians.sin()
var sti = ((catchmentArea.divide(22.13)).pow(0.6)).multiply(((sinSlope.divide(0.0896)).pow(1.3))).select(['b1'], ['sti']);

var aspect = ee.Terrain.aspect(elevation);

var hand = ee.ImageCollection('users/gena/global-hand/hand-100').mosaic().select(['b1'],['hand']).clip(region);

// =============================================================================================================

var dataset = ee.Image('USGS/NED');
var elevation = dataset.select('elevation');
var viz = {
  min: 108,
  max: 256,
  gamma: 1.6,
};
Map.addLayer(elevation.clip(region), viz, 'Elevation');
print('Elevation')
print(elevation.clip(region).getThumbURL({
  'min': 108, 'max': 256, 'gamma': 1.6,
  'format': 'png', 'dimensions': [891, 585]
}))


var viz = {
  min: 0.5,
  max: 3.5,
  palette: palettes.colorbrewer.Blues[9],
};
Map.setCenter(-100.55, 40.71, 5);
Map.addLayer(twi.clip(region), viz, 'slope');
print('Slope')
print(slope.clip(region).getThumbURL({
  'min': 0.5, 'max': 3.5, 'palette': palettes.colorbrewer.Blues[9],
  'format': 'png', 'dimensions': [891, 585]
}))


var viz = {
  min: 12,
  max: 32,
  palette: palettes.colorbrewer.YlGnBu[9],
};
Map.setCenter(-100.55, 40.71, 5);
Map.addLayer(twi.clip(region), viz, 'twi');
print('TWI')
print(twi.clip(region).getThumbURL({
  'min': 12, 'max': 32, 'palette': palettes.colorbrewer.YlGnBu[9],
  'format': 'png', 'dimensions': [891, 585]
}))

var viz = {
  min: 0,
  max: 2000,
  palette: palettes.colorbrewer.PuRd[9],
};
Map.setCenter(-100.55, 40.71, 5);
Map.addLayer(spi.clip(region), viz, 'spi');
print('SPI')
print(spi.clip(region).getThumbURL({
  'min': 0, 'max': 2000, 'palette': palettes.colorbrewer.PuRd[9],
  'format': 'png', 'dimensions': [891, 585]
}))

// var viz = {
//   min: 0,
//   max: 100,
//   palette: palettes.colorbrewer.YlOrRd[9],
// };
// Map.setCenter(-100.55, 40.71, 5);
// Map.addLayer(sti.clip(region), viz, 'sti');
// print('STI')
// print(sti.clip(region).getThumbURL({
//   'min': 0, 'max': 100, 'palette': palettes.colorbrewer.YlOrRd[9],
//   'format': 'png', 'dimensions': [891, 585]
// }))

var viz = {
  min: 0,
  max: 25,
  palette: palettes.colorbrewer.YlOrRd[9],
};
Map.setCenter(-100.55, 40.71, 5);
Map.addLayer(hand.clip(region), viz, 'hand');
print('HAND')
print(hand.clip(region).getThumbURL({
  'min': 0, 'max': 100, 'palette': palettes.colorbrewer.PuBuGn[9],
  'format': 'png', 'dimensions': [891, 585]
}))

var viz = {
  // min: 0,
  // max: 25,
  palette: palettes.matplotlib.viridis[7],
};
Map.setCenter(-100.55, 40.71, 5);
Map.addLayer(curve.clip(region), viz, 'curve');
print('Curve')
print(curve.clip(region).getThumbURL({
  'palette': palettes.matplotlib.viridis[7],
  'format': 'png', 'dimensions': [891, 585]
}))

var viz = {
  min: 0,
  max: 360,
  palette: palettes.matplotlib.plasma[7],
};
Map.setCenter(-100.55, 40.71, 5);
Map.addLayer(aspect.clip(region), viz, 'aspect');
print('Aspect')
print(aspect.clip(region).getThumbURL({
  'min': 0, 'max': 360, 'palette': palettes.matplotlib.plasma[7],
  'format': 'png', 'dimensions': [891, 585]
}))


// var dataset = ee.Image('CSP/ERGo/1_0/US/mTPI');
// var usMtpi = dataset.select('elevation');
// var usMtpiVis = {
//   min: -25,
//   max: 30,
//   palette: ['0b1eff', '4be450', 'fffca4', 'ffa011', 'ff0000'],
// };
// Map.addLayer(usMtpi.clip(region), usMtpiVis, 'US mTPI');
// print('US mTPI')
// print(mTPI.clip(region).getThumbURL({
//   'min': -25, 'max': 30, 'palette': ['0b1eff', '4be450', 'fffca4', 'ffa011', 'ff0000'],
//   'format': 'png', 'dimensions': [891, 585]
// }))


var dataset = ee.ImageCollection('USGS/NLCD');
var landcover = dataset.select('landcover');
var lulc_palette = [
    '000000', '000000', '000000', '000000', '000000', '000000', '000000', '000000', '000000', '000000', '000000', '466b9f', 'd1def8', '000000',
    '000000', '000000', '000000', '000000', '000000', '000000', '000000', 'dec5c5', 'd99282', 'eb0000','ab0000','000000', '000000', '000000',
    '000000', '000000', '000000',
    'b3ac9f',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '68ab5f',
    '1c5f2c',
    'b5c58f',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    'af963c',
    'ccb879',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    'dfdfc2',
    'd1d182',
    'a3cc51',
    '82ba9e',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    'dcd939',
    'ab6c28',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    '000000',
    'b8d9eb',
    '000000',
    '000000',
    '000000',
    '000000',
    '6c9fb8']

Map.addLayer(landcover.mosaic().clip(region),
{'min': 0.0, 'max': 95.0, 'palette': lulc_palette},
'Landcover'
);


Map.setCenter(ee.Number(centerpoint.coordinates().get(0)).getInfo(),
              ee.Number(centerpoint.coordinates().get(1)).getInfo(), 11)

print('LULC')
print(landcover.mosaic().clip(region).getThumbURL({
  'min': 0.0, 'max': 95.0, 'palette': lulc_palette,
  'format': 'png', 'dimensions': [891, 585]
}))


// Permanent Water Extent, Joint Research Centre, Global Surface Water dataset (JRC GSW)
var GSW = ee.Image('JRC/GSW1_0/GlobalSurfaceWater') // Permanent Water Extent, Joint Research Centre, Global Surface Water dataset (JRC GSW)
var occurrence = ee.Image('JRC/GSW1_0/GlobalSurfaceWater').select('occurrence')
var GSW_perm = occurrence.gte(40).rename('GSW_perm').unmask(0)
var GSW_perm = GSW_perm.fastDistanceTransform().rename('GSW_perm')  // Just for viz purposes for exporting
print('GSW perm')
print(GSW_perm.clip(region).getThumbURL({
  'min': 0, 'max': 1, 'palette': ['FFFFFF', '000000'],
  'format': 'png', 'dimensions': [891, 585]
}))

var GSW_seasonal = occurrence.gte(20).unmask(0)
var GSW_distSeasonal = GSW_seasonal.fastDistanceTransform().rename('GSW_distSeasonal') // Distance from permanent water bodies
print('GSW dist')
print(GSW_distSeasonal.clip(region).sqrt().getThumbURL({
  'min': 0, 'max': 40, 'palette': palettes.matplotlib.inferno[7].reverse(),
  'format': 'png', 'dimensions': [891, 585]
}))



