/* The function featuresFromID() uses the flow accumulation dataset from the high-resolution National
Hydrography Dataset, which is 10m spatial resolution. Those can be downloaded from the USGS and uploaded to GEE as assets:
https://www.usgs.gov/core-science-systems/ngp/national-hydrography/nhdplus-high-resolution. The other option is to use a
3 arcsecond (~90m) flow accumulation dataset that is commented below */

/* This filters an ImageCollection by date and bounds of DFO flood event */
function filterIC(ic, ft){
  var range = ee.DateRange(ft.get('Began'), ft.get('Ended'))
  var bounds = ft.geometry()
  var dfoID = ft.get('ID')
  return ic.filterBounds(bounds) // Event bounds
            .filterDate(range) // Date range
            .filterMetadata('CLOUD_COVER', 'less_than', 10)
            .map(function(image){
              return image.set('dfoID', dfoID);
            });
}

/* This identifies Landsat images that coincide with date range and extent of a flood event from the Dartmouth Flood
 Observatory. Input is flood event ID, type refers to Landsat product (surface reflectance, 'sr'; top of atmosphere, 'toa'
 */
function getLandsatImages(event, type){
  var dfoID = event.get('ID')
  if(type=='sr'){
    var l8 = filterIC(ee.ImageCollection('LANDSAT/LC08/C01/T1_SR'), event).toList(1000)
  } else if(type=='toa'){
    var l8 = filterIC(ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA'), event).toList(1000)
  }
  // var l7 = filterIC(ee.ImageCollection('LANDSAT/LE07/C01/T1_TOA'), event).toList(1000)
  // var l5 = filterIC(ee.ImageCollection('LANDSAT/LT05/C01/T1_TOA'), event).toList(1000)
  // var imageList = l8.add(l7).add(l5);
  var imageList = l8
  return ee.ImageCollection.fromImages(ee.List(imageList).flatten())
    .set({'dfoID': dfoID, // Add dfoID property to each image
    'Began': ee.Date(event.get('Began')),
    'Ended': ee.Date(event.get('Ended')).advance(3, 'day')});
}


/* This calculates the flood conditioning features for a Landsat 8 image when given the image ID */
exports.featuresFromID = function(imageID){

  var scene = ee.Image('LANDSAT/LC08/C01/T1_TOA/' + imageID)


  // -------------------------------------------------------------------
  // Detect inundation

  // Detect floods
  var feyisa = function(image){
    return image
    .expression("4 * (b('B3')-b('B6')) - (0.25 * b('B5') + 2.75 * b('B7'))")
    .gt(0);
  };

  // Compute the bits we need to extract.
  var getQABits = function(image, start, end, newName) {
      var pattern = 0;
      for (var i = start; i <= end; i++) {
         pattern += Math.pow(2, i);
      }
      // Return a single band image of the extracted QA bits, giving the band a new name.
      return image.select([0], [newName])
                    .bitwiseAnd(pattern)
                    .rightShift(start);
  };

  // A function to mask out cloud shadow pixels.
  var cloud_shadows = function(image) {
    // Select the QA band.
    var QA = image.select(['pixel_qa']);
    // Get the internal_cloud_algorithm_flag bit.
    // return getQABits(QA, 7,8, 'Cloud_shadows').eq(1); // if using BQA band of TOA
    return getQABits(QA, 3,3, 'Cloud_shadows').eq(0);
    // Return an image masking out cloudy areas.
  };

  // A function to mask out cloudy pixels.
  var clouds = function(image) {
    // Select the QA band.
    var QA = image.select(['pixel_qa']);
    // Get the internal_cloud_algorithm_flag bit.
    // return getQABits(QA, 4,4, 'Cloud').eq(0); // if using BQA band of TOA
    return getQABits(QA, 5,5, 'Cloud').eq(0);
    // Return an image masking out cloudy areas.
  };

  // A function to mask out snow/ice pixels.
  var snow = function(image){
      var QA = image.select(['pixel_qa'])
      // return getQABits(QA, 9, 10, 'snow_ice').eq(1) // if using BQA band of TOA
      return getQABits(QA, 4, 4, 'snow_ice').eq(0)
  }

  var QAscene = ee.Image('LANDSAT/LC08/C01/T1_SR/' + imageID)

  // Get SR product for pixel_qa band - BQA in TOA doesn't work properly
  var cs = cloud_shadows(QAscene);
  var c = clouds(QAscene);
  var s = snow(QAscene)

  var flooded = feyisa(scene).rename('flooded')
  flooded = flooded.updateMask(cs).updateMask(c).updateMask(s)

  // -------------------------------------------------------------------
  // Find drainage areas underlying flood event images

  // US watershed feature collection, HUC lvl 6
  var watershed = ee.FeatureCollection("ft:1jtXLmt6pJCaTmLuBIAE-CcuTPQX2SK72UPFsZ142")

  // Find drainage areas that intersect image
  function find_watersheds(ft, scene){
      var scene_bounds = scene.geometry()
      return ft.filterBounds(scene_bounds)
  }

  var watersheds = find_watersheds(watershed, scene)
  var region = scene.geometry()

  // -------------------------------------------------------------------
  // Flood conditioning factors

  // JRC permanent water extent
  var waterJRC = ee.Image.constant(0).blend(ee.Image('JRC/GSW1_0/GlobalSurfaceWater').select('occurrence').gte(50)).select(['constant'],['water'])

  // -------------------- Topographical features
  var mtpi = ee.Image('CSP/ERGo/1_0/US/mTPI').select(['elevation'], ['mtpi'])  // multi-scale topographic position index
  // var elevation = ee.Image("USGS/NED").select('elevation')
  var elevation = ee.Image("USGS/SRTMGL1_003").select('elevation')

  var slope = ee.Terrain.slope(elevation);

  // Curvature from https://github.com/zecojls/tagee
  var TAGEE = require('users/joselucassafanelli/TAGEE:TAGEE-functions');
  var bbox = scene.geometry()
  var hansen_2016 = ee.Image('UMD/hansen/global_forest_change_2016_v1_4').select('datamask');
  var hansen_2016_wbodies = hansen_2016.neq(1).eq(0);
  var waterMask = hansen_2016.updateMask(hansen_2016_wbodies);
  var demSRTM = elevation.clip(bbox).rename('DEM');
  // Smoothing filter
  var gaussianFilter = ee.Kernel.gaussian({radius: 3, sigma: 2, units: 'pixels', normalize: true});
  demSRTM = demSRTM.convolve(gaussianFilter).resample("bilinear")
  var DEMAttributes = TAGEE.terrainAnalysis(TAGEE, demSRTM, bbox).updateMask(waterMask);
  var curve = DEMAttributes.select('VerticalCurvature').unmask(0).rename('curve')

/* Use the second flowAcc if using HRNHDPlus dataset. Must download your own, upload as asset to GEE, and link to it here
    Otherwise, use the first flowAcc */
//   var flowAcc = ee.ImageCollection("users/imerg/flow_acc_3s").mosaic() // 3 arcseconds
  var flowAcc = ee.ImageCollection("users/ianpdavies/HRNHDPlus_FA")
        .map(function(img) {return img.toInt32()})
        .mosaic()
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

  // Sediment transport index
  var sinSlope = slopeRadians.sin()
  var sti = ((catchmentArea.divide(22.13)).pow(0.6)).multiply(((sinSlope.divide(0.0896)).pow(1.3))).select(['b1'], ['sti']);

  var aspect = ee.Terrain.aspect(elevation);

  var hand = ee.ImageCollection('users/gena/global-hand/hand-100').mosaic().select(['b1'],['hand']).clip(region);

  // -------------------- Other features

  // Permanent Water Extent, Joint Research Centre, Global Surface Water dataset (JRC GSW)
  var GSW = ee.Image('JRC/GSW1_0/GlobalSurfaceWater') // Permanent Water Extent, Joint Research Centre, Global Surface Water dataset (JRC GSW)
  var occurrence = ee.Image('JRC/GSW1_0/GlobalSurfaceWater').select('occurrence')

  var GSWPerm = occurrence.gte(40).rename('GSWPerm').unmask(0)
  var GSW_seasonal = occurrence.gte(20).unmask(0)

  // Distance from permanent water extent. Probably don't need both of these, might just use extent
  var GSWDistSeasonal = GSW_seasonal.fastDistanceTransform().rename('GSWDistSeasonal') // Distance from permanent water bodies
  // var GSW_distExtent = GSW_maxExtent.fastDistanceTransform().rename('GSW_distExtent') // Distance from max extent of water

  // Remove flooded pixels that are part of permanent water extent
  // flooded = flooded.updateMask(GSW_perm.not()).unmask(0)

  // US lithology
  var lith = ee.Image("CSP/ERGo/1_0/US/lithology").select(['b1'], ['lith'])

  // NLCD: National Land Cover Dataset
  // var imp = ee.Image('ESA/GLOBCOVER_L4_200901_200912_V2_3').select(0).eq(190).select(['landcover'],['impervious']).clip(region);
  var nlcd = ee.Image('USGS/NLCD/NLCD2011')

  // // Projection needs to be the same as the other bands, otherwise the stacked output geometry is slightly off
  nlcd = nlcd
    // .unitScale(-2000, 10000)
    .reproject('EPSG:4326', null, 30);

  var landcover = nlcd.select(0)

  // -------------------------------------------------------------------
  // Reduce resolution to 30m to match Landsat

  spi = spi
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 512,
      bestEffort: true})
    .reproject(aspect.projection());

  sti = sti
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 512,
      bestEffort: true})
    .reproject(aspect.projection());

  twi = twi
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 512,
      bestEffort: true})
    .reproject(aspect.projection());

  // -------------------------------------------------------------------
  // Structuring data and sampling for classifier training

  // Combine all features into one image, each feature one band

  scene = scene.select(['B1','B2','B3','B4','B5','B6','B7'])

  var features = (scene
    .addBands(elevation)
    .addBands(slope)
    .addBands(curve)
    .addBands(landcover) // Can disaggregate to dummy features in Python
    .addBands(twi)
    .addBands(spi)
    .addBands(sti)
    .addBands(mtpi)
    .addBands(lith)
    .addBands(GSWPerm)
    .addBands(GSWDistSeasonal)
    .addBands(aspect)
    .addBands(hand)
    .addBands(flooded))
    .updateMask(c).updateMask(cs).updateMask(s) // This bandaids the problem with small QA band extent
    .clip(scene.geometry());

  features = features.reproject({crs: aspect.projection().crs(), scale: 30})

  return features
}

