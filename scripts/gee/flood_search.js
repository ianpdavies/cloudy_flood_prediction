/* Find and examine Landsat images that cover a given Dartmouth Flood Observatory event */

var table = ee.FeatureCollection("users/ianpdavies/DFO_Database"),

// ----------------------------------------------------------------------

var eventID = 4510

/* Search all DFO events, then select images by DFO ID after
 the import feature collection with flood events */
var feats = ee.FeatureCollection('ft:1uQgyUakR1ZiXeKZNc2uay69A_5vJe0iC882-Id8E');

// Set the end date 3 days after the begin date and make them ee.Dates
var dfo = feats.map(function(event){
  var newBegan = ee.Date(event.get('Began'))
  var newEnded = ee.Date(event.get('Ended')).advance(3, 'day')
  return (event.set({'Began': newBegan, 'Ended': newEnded}))
})

// For now, restrict to only Landsat 8 imagery (launched Feb 11, 2013)
dfo = dfo.filterMetadata('Began', 'greater_than', ee.Date('2013-02-11'))

// Restrict to event of interest
dfo = ee.FeatureCollection(dfo.filterMetadata('ID', 'equals', eventID))

Map.addLayer(dfo.draw('red'), {}, 'DFO Event')

// Function to filter the ImageCollection by date and bounds of flood event
function filterIC(ic, ft){
  var range = ee.DateRange(ft.get('Began'), ft.get('Ended'))
  var bounds = ft.geometry()
  var dfoID = ft.get('ID')
  return ic.filterBounds(bounds).filterDate(range).map(function(image){
    return image.set('dfoID', dfoID);
  });
}

// Get Landsat images
function getLandsatImages(event){
  var dfoID = event.get('ID')
  var l8 = filterIC(ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA'), event).toList(1000)
  // var l8 = filterIC(ee.ImageCollection('LANDSAT/LC08/C01/T1_SR'), event).toList(1000)
  // var l7 = filterIC(ee.ImageCollection('LANDSAT/LE07/C01/T1_TOA'), event).toList(1000)
  // var l5 = filterIC(ee.ImageCollection('LANDSAT/LT05/C01/T1_TOA'), event).toList(1000)
  // var imageList = l8.add(l7).add(l5);
  var imageList = l8
  return ee.ImageCollection.fromImages(ee.List(imageList).flatten())
    .set({'dfoID': dfoID,
    'Began': ee.Date(event.get('Began')),
    'Ended': ee.Date(event.get('Ended')).advance(3, 'day')});
}

var imgs = ee.ImageCollection(dfo.map(getLandsatImages)).flatten();
print(imgs)
print(ee.Date(dfo.first().get('Began')))
print(ee.Date(dfo.first().get('Ended')))

// ----------------------------------------------------------------------
// Process images

// Water Masks
var GSW = ee.Image('JRC/GSW1_0/GlobalSurfaceWater') // Permanent Water Extent, Joint Research Centre, Global Surface Water dataset (JRC GSW)
var GSW_trans = GSW.select(5) // Band 5 ("transition")
var transmask = GSW_trans.unmask(0).not()
var GSW_perm = ee.Image.constant(0).blend(ee.Image('JRC/GSW1_0/GlobalSurfaceWater').select('occurrence').gte(50)).select(['constant'],['water'])
GSW_perm.rename('GSW_perm')

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

// Add all L8 images to map
var s2 = imgs
function addImage(image) { // display each image in collection
  var img = ee.Image(image.id)
  var id = img.id().getInfo()
  print(id)
  Map.addLayer(img, {bands: ['B4','B3','B2'], min: 0.04, max: 0.2}, 'RGB - '+ id, false)
  Map.addLayer(img, {bands: ['B5','B4','B3'], min: 0.03, max: 0.3}, 'CIR - ' + id, false)
  img = getFloods(img).updateMask(GSW_perm.not()).selfMask()
  Map.addLayer(img, {palette:['black','red']}, 'Flood - ' + id, false)
}

s2.evaluate(function(s2) {  // use map on client-side
  s2.features.map(addImage)
})


Map.centerObject(dfo.geometry().centroid(), 4)

// Permanent water
var GSW_perm = ee.Image.constant(0).blend(ee.Image('JRC/GSW1_0/GlobalSurfaceWater').select('occurrence').gte(50)).select(['constant'],['water'])
Map.addLayer(GSW_perm.selfMask(), {palette:['lightblue']}, 'perm water', false)