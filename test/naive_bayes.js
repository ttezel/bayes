var assert = require('assert')
  , fs = require('fs')
  , path = require('path')
  , bayes = require('../lib/naive_bayes')

describe('naive bayes classifier', function () {

  //sentiment analysis test
  it('categorizes correctly for `positive` and `negative` categories', function (done) {

    var classifier = bayes()

    //teach it positive phrases
    classifier.learn('amazing, awesome movie!! Yeah!!', 'positive')
    classifier.learn('Sweet, this is incredibly, amazing, perfect, great!!', 'positive')

    //teach it a negative phrase
    classifier.learn('terrible, shitty thing. Damn. Sucks!!', 'negative')

    //teach it a neutral phrase
    classifier.learn('I dont really know what to make of this.', 'neutral')

    //now test it to see that it correctly categorizes a new document
    assert.equal(classifier.categorize('awesome, cool, amazing!! Yay.'), 'positive')
    done()
  })

  //topic analysis test
  it('categorizes correctly for `chinese` and `japanese` categories', function (done) {

    var classifier = bayes()

    //teach it how to identify the `chinese` category
    classifier.learn('Chinese Beijing Chinese', 'chinese')
    classifier.learn('Chinese Chinese Shanghai', 'chinese')
    classifier.learn('Chinese Macao', 'chinese')

    //teach it how to identify the `japanese` category
    classifier.learn('Tokyo Japan Chinese', 'japanese')

    //make sure it learned the `chinese` category correctly
    var chineseFrequencyCount = classifier.wordFrequencyCount.chinese

    assert.equal(chineseFrequencyCount['Chinese'], 5)
    assert.equal(chineseFrequencyCount['Beijing'], 1)
    assert.equal(chineseFrequencyCount['Shanghai'], 1)
    assert.equal(chineseFrequencyCount['Macao'], 1)

    //make sure it learned the `japanese` category correctly
    var japaneseFrequencyCount = classifier.wordFrequencyCount.japanese

    assert.equal(japaneseFrequencyCount['Tokyo'], 1)
    assert.equal(japaneseFrequencyCount['Japan'], 1)
    assert.equal(japaneseFrequencyCount['Chinese'], 1)

    //now test it to see that it correctly categorizes a new document
    assert.equal(classifier.categorize('Chinese Chinese Chinese Tokyo Japan'), 'chinese')

    done()
  })

  it('serializes/deserializes its state as JSON correctly.', function (done) {
    var classifier = bayes()

    classifier.learn('Fun times were had by all', 'positive')
    classifier.learn('sad dark rainy day in the cave', 'negative')

    var jsonRepr = classifier.toJson()

    // check serialized values
    var state = JSON.parse(jsonRepr)

    // ensure classifier's state values are all in the json representation
    bayes.STATE_KEYS.forEach(function (k) {
      assert.deepEqual(state[k], classifier[k])
    })

    var revivedClassifier = bayes.fromJson(jsonRepr)

    // ensure the revived classifier's state is same as original state
    bayes.STATE_KEYS.forEach(function (k) {
      assert.deepEqual(revivedClassifier[k], classifier[k])
    })

    done()
  })

})
