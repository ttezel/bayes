"use strict";

const STATE_KEYS = [
  'categories', 'docCount', 'totalDocuments', 'vocabulary', 'vocabularySize',
  'wordCount', 'wordFrequencyCount', 'options'
];

function fromJson(jsonStr) {
  let parsed;

  try {
    parsed = JSON.parse(jsonStr);
  } catch(e) {
    throw new Error('NaiveBayes.fromJson expects a valid JSON string.');
  }

  // init a new classifier
  let classifier = new NaiveBayes(parsed.options);

  // override the classifier's state
  STATE_KEYS.forEach(key => {
    if (!parsed[key]) {
      throw new Error(`NaiveBayes.fromJson: JSON string is missing an expected property: ${key}.`);
    }
    classifier[key] = parsed[key];
  });

  return classifier;
};

function defaultTokenizer(text) {
  // remove punctuation from text - remove anything that isn't a word char or a space
  let rgxPunctuation = /[^\w\s]/g;

  let sanitized = text.replace(rgxPunctuation, ' ');

  return sanitized.split(/\s+/);
}

class NaiveBayes {
  constructor(options) {
    this.options = {};

    if (typeof options !== 'undefined') {
      if (!options || typeof options !== 'object' || Array.isArray(options)) {
        throw TypeError(`NaiveBayes got invalid 'options': '${options}'. Pass in an object.`);
      }
      this.options = options;
    }

    // use defaultTokenizer unless one was passed in
    this.tokenizer = this.options.tokenizer || defaultTokenizer;

    // initialize our vocabulary and its size
    this.vocabulary = {};
    this.vocabularySize = 0;

    // number of documents we have learned from
    this.totalDocuments = 0;

    // document frequency table for each of our categories
    // => for each category, how often were documents mapped to it
    this.docCount = {};

    // for each category, how many words total were mapped to it
    this.wordCount = {};

    // word frequency table for each category
    // => for each category, how frequent was a given word mapped to it
    this.wordFrequencyCount = {};

    // hashmap of our category names
    this.categories = {};
  }

  initializeCategory(categoryName) {
    if (!this.categories[categoryName]) {
      this.docCount[categoryName] = 0;
      this.wordCount[categoryName] = 0;
      this.wordFrequencyCount[categoryName] = {};
      this.categories[categoryName] = true;
    }
    return this;
  }

  learn(text, category) {
    // initialize category data structures if we've never seen this category
    this.initializeCategory(category);

    // update our count of how many documents mapped to this category
    this.docCount[category]++;

    // update the total number of documents we have learned from
    this.totalDocuments++;

    // normalize the text into a word Array
    let tokens = this.tokenizer(text);

    // get a frequency count for each token in the text
    let frequencyTable = this.frequencyTable(tokens);

    // update our vocabulary and our word frequency cound for this category
    for (let token in frequencyTable) {
      // add this word to our vocabulary if not already existing
      if (!this.vocabulary[token]) {
        this.vocabulary[token] = true;
        this.vocabularySize++;
      }

      let frequencyInText = frequencyTable[token];

      // update the frequency information for this word in this category
      if (!this.wordFrequencyCount[category][token]) {
        this.wordFrequencyCount[category][token] = frequencyInText;
      } else {
        this.wordFrequencyCount[category][token] += frequencyInText;
      }

      // update the count of all words we have seen mapped in this category
      this.wordCount[category] += frequencyInText;
    }

    return this;
  }

  categorize(text) {
    let maxProbability = Number.NEGATIVE_INFINITY,
        chosenCategory;

    let tokens = this.tokenizer(text);
    let frequencyTable = this.frequencyTable(tokens);

    // iterate thru our categories to find the one with max probability for this text
    for (let category in this.categories) {
      // start by calculating the overall probability of this category
      // => out of all documents we've ever looked at, how many were
      // mapped to this category
      let categoryProbability = this.docCount[category] / this.totalDocuments;

      // take the log to avoid underflow
      let logProbability = Math.log(categoryProbability);

      for (let token in frequencyTable) {
        let frequencyInText = frequencyTable[token];
        let tokenProbability = this.tokenProbability(token, category);

        // determine the log of the P( w | c ) for this word
        logProbability += frequencyInText * Math.log(tokenProbability);
      }

      if (logProbability > maxProbability) {
        maxProbability = logProbability;
        chosenCategory = category;
      }
    }

    return chosenCategory;
  }

  tokenProbability(token, category) {
    // how many times this word has occurred in documents mapped to this category
    let wordFrequencyCount = this.wordFrequencyCount[category][token] || 0;

    // what is the count of all words that have ever been mapped to this category
    let wordCount = this.wordCount[category];

    // use laplace Add-1 Smoothing equation
    return (wordFrequencyCount + 1) / (wordCount + this.vocabularySize);
  }

  frequencyTable(tokens) {
    let frequencyTable = {};

    tokens.forEach(token => {
      if (!frequencyTable[token]) {
        frequencyTable[token] = 1;
      } else {
        frequencyTable[token]++;
      }
    });

    return frequencyTable;
  }

  toJson() {
    let state = {};

    STATE_KEYS.forEach(key => {
      state[key] = this[key];
    });

    let jsonStr = JSON.stringify(state);

    return jsonStr;
  }
}

module.exports = function(options) {
  return new NaiveBayes(options);
}

module.exports.STATE_KEYS = STATE_KEYS;
module.exports.fromJson = fromJson;
