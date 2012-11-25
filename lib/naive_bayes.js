/*
    Expose our naive-bayes generator function
 */
module.exports = function (options) {
  return new Naivebayes(options)
}

/**
 * Naive-Bayes Classifier
 *
 * This is a naive-bayes classifier that uses Laplace Smoothing.
 *
 * usage: 
 * 
 * new Naivebayes({
 *     categories: [ 'positive', 'negative' ]
 * })
 *
 * Takes an (optional) options object containing:
 *   - `categories` => array of categories to classify documents with.
 * 
 */
function Naivebayes (options) {
  if (!options) options = {}

  //make sure we have an array of categories
  if (!options.categories)
    options.categories = []
  else if (!Array.isArray(options.categories))
    throw new Error('`categories` must be an array')

  //initialize our vocabulary and its size
  this.vocabulary = {}
  this.vocabularySize = 0

  //number of documents we have learned from
  this.totalDocuments = 0

  //document frequency table for each of our categories
  //=> for each category, how often were documents mapped to it
  this.docCount = {}

  //for each category, how many words total were mapped to it
  this.wordCount = {}

  //word frequency table for each category
  //=> for each category, how frequent was a given word mapped to it
  this.wordFrequencyCount = {}

  //hash of our category names
  this.categories = {}

  //initialize our category data structures
  options.categories.forEach(this.initializeCategory.bind(this))
}

/**
 * Initialize each of our data structure entries for this new category
 * 
 * @param  {String} categoryName
 */
Naivebayes.prototype.initializeCategory = function (categoryName) {
  if (!this.categories[categoryName]) {
    this.docCount[categoryName] = 0
    this.wordCount[categoryName] = 0
    this.wordFrequencyCount[categoryName] = {}
    this.categories[categoryName] = true
  }
  return this
}

/**
 * train our naive-bayes classifier by telling it what `category`
 * the `text` corresponds to.
 * 
 * @param  {String} text    
 * @param  {String} class 
 */
Naivebayes.prototype.learn = function (text, category) {
  var self = this
  
  //initialize category data structures if we've never seen this category
  self.initializeCategory(category)

  //update our count of how many documents mapped to this category
  self.docCount[category]++

  //update the total number of documents we have learned from
  self.totalDocuments++

  //normalize the text into a word array
  var tokens = self.tokenize(text)

  //get a frequency count for each token in the text
  var frequencyTable = self.frequencyTable(tokens)

  /*
      Update our vocabulary and our word frequency count for this category
   */
  
  Object
  .keys(frequencyTable)
  .forEach(function (token) {
    //add this word to our vocabulary if not already existing
    if (!self.vocabulary[token]) {
      self.vocabulary[token] = true
      self.vocabularySize++
    }

    var frequencyInText = frequencyTable[token]

    //update the frequency information for this word in this category
    if (!self.wordFrequencyCount[category][token])
      self.wordFrequencyCount[category][token] = frequencyInText
    else
      self.wordFrequencyCount[category][token] += frequencyInText

    //update the count of all words we have seen mapped to this category
    self.wordCount[category] += frequencyInText
  })

  return self
}

/**
 * Determine what category `text` belongs to.
 * 
 * @param  {String} text 
 * @return {String} category
 */
Naivebayes.prototype.categorize = function (text) {
  var self = this
    , maxProbability = -Infinity
    , chosenCategory = null
  
  var tokens = self.tokenize(text)
  var frequencyTable = self.frequencyTable(tokens)

  //iterate thru our categories to find the one with max probability for this text
  Object
  .keys(self.categories)
  .forEach(function (category) {

    //start by calculating the overall probability of this category
    //=>  out of all documents we've ever looked at, how many were 
    //    mapped to this category
    var categoryProbability = self.docCount[category] / self.totalDocuments

    //take the log to avoid underflow
    var logProbability = Math.log(categoryProbability)

    //now determine P( w | c ) for each word `w` in the text
    Object
    .keys(frequencyTable)
    .forEach(function (token) {
      var frequencyInText = frequencyTable[token]
      var tokenProbability = self.tokenProbability(token, category)

      //console.log('token: %s category: `%s` tokenProbability: %d', token, category, tokenProbability)

      //determine the log of the P( w | c ) for this word
      logProbability += frequencyInText * Math.log(tokenProbability)
    })

    if (logProbability > maxProbability) {
      maxProbability = logProbability
      chosenCategory = category
    }
  })

  return chosenCategory
}

/**
 * Calculate probability that a `token` belongs to a `category`
 * 
 * @param  {String} token    
 * @param  {String} category 
 * @return {Number} probability
 */
Naivebayes.prototype.tokenProbability = function (token, category) {
  //how many times this word has occurred in documents mapped to this category
  var wordFrequencyCount = this.wordFrequencyCount[category][token] || 0

  //what is the count of all words that have ever been mapped to this category
  var wordCount = this.wordCount[category]

  //use laplace Add-1 Smoothing equation
  return ( wordFrequencyCount + 1 ) / ( wordCount + this.vocabularySize )
}

/**
 * Build a frequency hash where 
 * - the keys are the words in `text`
 * - the values are the frequency of the word in `text`
 * 
 * @param  {Array} tokens  Normalized word array 
 * @return {Object}
 */
Naivebayes.prototype.frequencyTable = function (tokens) {
  var frequencyTable = {}

  tokens.forEach(function (token) {
    if (!frequencyTable[token])
      frequencyTable[token] = 1
    else
      frequencyTable[token]++
  })

  return frequencyTable
}

/**
 * Given an input string, tokenize it into an array of word tokens
 * 
 * @param  {String} text
 * @return {Array}
 */
Naivebayes.prototype.tokenize = function (text) {
  //remove punctuation from text - remove anything that isn't a word char or a space
  var rgxPunctuation = /[^\w\s]/g

  var sanitized = text.replace(rgxPunctuation, ' ')

  return sanitized.split(/\s+/)
}

