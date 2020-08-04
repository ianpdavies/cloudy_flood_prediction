/* Examine images (which can be identified using flood_search.js) for visible flooding and subset using the geometry.
The geometry, dfoID, batch, and imageID are required for export_to_drive.js. Smaller subsets make modeling more manageable
 in Python */

var geometry = /* color: #0b4a8b */ee.Geometry.Polygon(
        [[[-121.2280558, 36.85877823],
          [-120.3189371, 36.85877823],
          [-120.3189371, 37.66102516],
          [-121.2280558, 37.66102516],
          [-121.2280558, 36.85877823]]]);

//======================================================================
var imageID = 'LC08_043034_20170303'
var dfoID = 4444
var batch = 1
var img = ee.Image('LANDSAT/LC08/C01/T1_TOA/'+imageID)

// ----------------------------------------------------------------------
// Process images

// Water Masks
var GSW = ee.Image('JRC/GSW1_0/GlobalSurfaceWater') // Permanent Water Extent, Joint Research Centre, Global Surface Water dataset (JRC GSW)
var GSW_trans = GSW.select(5) // Band 5 ("transition")
var transmask = GSW_trans.unmask(0).not()
var waterJRC = ee.Image.constant(0).blend(ee.Image('JRC/GSW1_0/GlobalSurfaceWater').select('occurrence').gte(50)).select(['constant'],['water'])


// Detect water and remove cloudy/snowy pixels
function getFloods(image){
  // Detect floods
  var feyisa = function(image){
    return image
    .expression("4 * (b('B3')-b('B6')) - (0.25 * b('B5') + 2.75 * b('B7'))")
    .gt(0);
  };

  // Cloud/shadow mask code from https://gis.stackexchange.com/questions/274612/apply-a-cloud-mask-to-a-landsat8-collection-in-google-earth-engine-time-series

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

  var id = ee.Image(image).id().getInfo()
  var sliceIndex = ee.Number(ee.String(id).rindex('LC'))
  id = ee.String(ee.String(id).slice(sliceIndex)).getInfo()
  var QAscene = ee.Image('LANDSAT/LC08/C01/T1_SR/' + id)

  // Get SR product for pixel_qa band - BQA in TOA doesn't work properly
  var cs = cloud_shadows(QAscene);
  var c = clouds(QAscene);
  var s = snow(QAscene)

  var flooded = feyisa(ee.Image(image)).rename('flooded')
  flooded = flooded.updateMask(cs).updateMask(c).updateMask(s)
  return flooded
}

// ----------------------------------------------------------------------
// Visualize images

Map.addLayer(img, {bands: ['B4','B3','B2'], min: 0.04, max: 0.2}, 'RGB', true)
Map.addLayer(img, {bands: ['B5','B4','B3'], min: 0.03, max: 0.3}, 'CIR', false)
var floods = getFloods(img).updateMask(waterJRC.not()).selfMask()
Map.addLayer(floods, {palette:['black','red']}, 'Flood', true)


Map.centerObject(img.geometry().centroid(), 8)