// Global variables
let model;
let isTraining = false;

// Initialize application when window loads
window.onload = async function() {
    try {
        // Update status
        updateStatus('Loading MovieLens data...');
        
        // Load data first
        await loadData();
        
        // Populate dropdowns
        populateUserDropdown();
        populateMovieDropdown();
        
        // Update status and start training
        updateStatus('Data loaded. Training model...');
        
        // Train the model
        await trainModel();
        
    } catch (error) {
        console.error('Initialization error:', error);
        updateStatus('Error initializing application: ' + error.message, true);
    }
};

function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '';
    
    // Get unique user IDs from the ratings data
    const userIds = [...new Set(ratings.map(r => r.userId))];
    
    // Add users to the dropdown
    userIds.forEach(userId => {
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = `User ${userId}`;
        userSelect.appendChild(option);
    });
}

function populateMovieDropdown() {
    const movieSelect = document.getElementById('movie-select');
    movieSelect.innerHTML = '';
    
    // Add movies
    movies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.year ? `${movie.title} (${movie.year})` : movie.title;
        movieSelect.appendChild(option);
    });
}

async function trainModel() {
    try {
        isTraining = true;
        
        // Define model architecture with better configuration
        const numLatentFactors = 10; // Slightly more factors for better discrimination
        
        // User pathway
        const userInput = tf.layers.input({ shape: [1], name: 'user-input' });
        const userEmbedding = tf.layers.embedding({
            inputDim: numUsers + 1,
            outputDim: numLatentFactors,
            embeddingsRegularizer: tf.regularizers.l2({ l2: 0.001 }), // Regularization to prevent uniform predictions
            name: 'user-embedding'
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        
        // User bias term - helps differentiate user rating tendencies
        const userBias = tf.layers.embedding({
            inputDim: numUsers + 1,
            outputDim: 1,
            embeddingsRegularizer: tf.regularizers.l2({ l2: 0.001 }),
            name: 'user-bias'
        }).apply(userInput);
        const userBiasFlatten = tf.layers.flatten().apply(userBias);
        
        // Movie pathway
        const movieInput = tf.layers.input({ shape: [1], name: 'movie-input' });
        const movieEmbedding = tf.layers.embedding({
            inputDim: numMovies + 1,
            outputDim: numLatentFactors,
            embeddingsRegularizer: tf.regularizers.l2({ l2: 0.001 }), // Regularization to prevent uniform predictions
            name: 'movie-embedding'
        }).apply(movieInput);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
        // Movie bias term - helps differentiate movie popularity
        const movieBias = tf.layers.embedding({
            inputDim: numMovies + 1,
            outputDim: 1,
            embeddingsRegularizer: tf.regularizers.l2({ l2: 0.001 }),
            name: 'movie-bias'
        }).apply(movieInput);
        const movieBiasFlatten = tf.layers.flatten().apply(movieBias);
        
        // Dot product of user and movie embeddings
        const dotProduct = tf.layers.dot({ axes: 1 }).apply([userFlatten, movieFlatten]);
        
        // Combine dot product with bias terms
        const sum = tf.layers.add().apply([dotProduct, userBiasFlatten, movieBiasFlatten]);
        
        // Scale output to 1-5 range using sigmoid activation
        // Sigmoid outputs 0-1, we scale to 1-5: output = 1 + 4 * sigmoid(x)
        const scaledOutput = tf.layers.lambda({
            fn: (x) => {
                // Apply sigmoid to get 0-1 range, then scale to 1-5 range
                const sigmoid = tf.sigmoid(x);
                return tf.add(tf.mul(sigmoid, 4), 1);
            },
            name: 'scale-to-rating'
        }).apply(sum);
        
        model = tf.model({ inputs: [userInput, movieInput], outputs: scaledOutput });
        
        // Use a lower learning rate for more stable training
        const optimizer = tf.train.adam(0.001);
        model.compile({ optimizer: optimizer, loss: 'meanSquaredError' });

        // Prepare data for training
        const userTensors = tf.tensor2d(ratings.map(r => r.userId), [ratings.length, 1]);
        const movieTensors = tf.tensor2d(ratings.map(r => r.movieId), [ratings.length, 1]);
        const ratingTensors = tf.tensor2d(ratings.map(r => r.rating), [ratings.length, 1]);

        // Train the model with more epochs for better convergence
        await model.fit([userTensors, movieTensors], ratingTensors, {
            epochs: 8, // Slightly more epochs
            batchSize: 128, // Larger batch size for stability
            validationSplit: 0.1, // Add validation to monitor overfitting
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const valLoss = logs.val_loss ? `, Val Loss: ${logs.val_loss.toFixed(4)}` : '';
                    updateStatus(`Epoch ${epoch + 1}/8, Loss: ${logs.loss.toFixed(4)}${valLoss}`);
                }
            }
        });
        
        // Clean up tensors
        tf.dispose([userTensors, movieTensors, ratingTensors]);

        updateStatus('Model training complete!');
        document.getElementById('predict-btn').disabled = false;
        
    } catch (error) {
        console.error('Training error:', error);
        updateStatus('Error training model: ' + error.message, true);
    } finally {
        isTraining = false;
    }
}

async function predictRating() {
    if (!model || isTraining) return;

    try {
        const userId = parseInt(document.getElementById('user-select').value);
        const movieId = parseInt(document.getElementById('movie-select').value);

        if (isNaN(userId) || isNaN(movieId)) {
            updateResult('Please select a user and a movie.', 'low');
            return;
        }

        // Create input tensors
        const userTensor = tf.tensor2d([userId], [1, 1]);
        const movieTensor = tf.tensor2d([movieId], [1, 1]);
        
        // Make prediction
        const prediction = model.predict([userTensor, movieTensor]);
        const rating = await prediction.data();
        const predictedRating = rating[0];
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, prediction]);
        
        // Display result
        const movie = movies.find(m => m.id === movieId);
        const movieTitle = movie ? (movie.year ? `${movie.title} (${movie.year})` : movie.title) : `Movie ${movieId}`;
        
        let ratingClass = 'medium';
        if (predictedRating >= 4) ratingClass = 'high';
        else if (predictedRating <= 2) ratingClass = 'low';
        
        updateResult(
            `Predicted rating for User ${userId} on \"${movieTitle}\": <strong>${predictedRating.toFixed(2)}/5</strong>`,
            ratingClass
        );
        
    } catch (error) {
        console.error('Prediction error:', error);
        updateResult('Error making prediction: ' + error.message, 'low');
    }
}

// New function to analyze user prediction patterns
async function analyzeUserPatterns() {
    if (!model) return;
    
    try {
        const userId = parseInt(document.getElementById('user-select').value);
        if (isNaN(userId)) return;
        
        // Sample some movies to check prediction variance
        const sampleMovies = movies.slice(0, 50).map(m => m.id); // Sample first 50 movies
        
        const predictions = [];
        for (const movieId of sampleMovies) {
            const userTensor = tf.tensor2d([userId], [1, 1]);
            const movieTensor = tf.tensor2d([movieId], [1, 1]);
            const prediction = model.predict([userTensor, movieTensor]);
            const rating = await prediction.data();
            predictions.push(rating[0]);
            
            tf.dispose([userTensor, movieTensor, prediction]);
        }
        
        // Calculate statistics
        const min = Math.min(...predictions);
        const max = Math.max(...predictions);
        const range = max - min;
        
        console.log(`User ${userId} prediction range: ${range.toFixed(2)} (${min.toFixed(2)}-${max.toFixed(2)})`);
        
    } catch (error) {
        console.error('Analysis error:', error);
    }
}

// UI helper functions
function updateStatus(message, isError = false) {
    const statusElement = document.getElementById('status');
    statusElement.textContent = message;
    statusElement.style.borderLeftColor = isError ? '#e74c3c' : '#3498db';
    statusElement.style.background = isError ? '#fdedec' : '#f8f9fa';
}

function updateResult(message, className = '') {
    const resultElement = document.getElementById('result');
    resultElement.innerHTML = message;
    resultElement.className = `result ${className}`;
}