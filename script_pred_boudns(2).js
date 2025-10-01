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
        
        // Simple and effective model architecture
        const userInput = tf.input({shape: [1]});
        const movieInput = tf.input({shape: [1]});
        
        // User embedding with regularization to prevent uniform predictions
        const userEmbedding = tf.layers.embedding({
            inputDim: numUsers + 1,
            outputDim: 10,
            embeddingsInitializer: 'randomNormal',
            embeddingsRegularizer: tf.regularizers.l2({l2: 0.01})
        }).apply(userInput);
        
        // Movie embedding with regularization
        const movieEmbedding = tf.layers.embedding({
            inputDim: numMovies + 1,
            outputDim: 10,
            embeddingsInitializer: 'randomNormal', 
            embeddingsRegularizer: tf.regularizers.l2({l2: 0.01})
        }).apply(movieInput);
        
        // Flatten embeddings
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
        // Dot product
        const dot = tf.layers.dot({axes: -1}).apply([userFlatten, movieFlatten]);
        
        // Add bias terms to increase prediction diversity
        const userBias = tf.layers.embedding({
            inputDim: numUsers + 1,
            outputDim: 1
        }).apply(userInput);
        const movieBias = tf.layers.embedding({
            inputDim: numMovies + 1, 
            outputDim: 1
        }).apply(movieInput);
        
        const userBiasFlat = tf.layers.flatten().apply(userBias);
        const movieBiasFlat = tf.layers.flatten().apply(movieBias);
        
        // Combine everything
        const combined = tf.layers.add().apply([dot, userBiasFlat, movieBiasFlat]);
        
        // SCALE OUTPUT TO 1-5 RANGE - This is the key fix
        const scaledOutput = tf.layers.lambda({
            function: (x) => {
                // Use tanh to get -1 to 1 range, then scale to 1-5
                return tf.tanh(x).mul(2).add(3);
            }
        }).apply(combined);
        
        model = tf.model({
            inputs: [userInput, movieInput],
            outputs: scaledOutput
        });
        
        // Compile model
        model.compile({
            optimizer: 'adam',
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        // Prepare training data
        const userTensor = tf.tensor1d(ratings.map(r => r.userId));
        const movieTensor = tf.tensor1d(ratings.map(r => r.movieId)); 
        const ratingTensor = tf.tensor1d(ratings.map(r => r.rating));

        // Train model
        await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: 10,
            batchSize: 32,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    updateStatus(`Epoch ${epoch + 1}/10 - Loss: ${logs.loss.toFixed(4)}, Val Loss: ${logs.val_loss.toFixed(4)}`);
                }
            }
        });

        // Cleanup
        tf.dispose([userTensor, movieTensor, ratingTensor]);

        updateStatus('Model training complete! Ready for predictions.');
        document.getElementById('predict-btn').disabled = false;
        
    } catch (error) {
        console.error('Training error:', error);
        updateStatus('Error training model: ' + error.message, true);
    } finally {
        isTraining = false;
    }
}

async function predictRating() {
    if (!model || isTraining) {
        updateResult('Model not ready. Please wait for training to complete.', 'low');
        return;
    }

    try {
        const userId = parseInt(document.getElementById('user-select').value);
        const movieId = parseInt(document.getElementById('movie-select').value);

        if (isNaN(userId) || isNaN(movieId)) {
            updateResult('Please select both a user and a movie.', 'low');
            return;
        }

        // Create tensors
        const userTensor = tf.tensor1d([userId]);
        const movieTensor = tf.tensor1d([movieId]);
        
        // Make prediction
        const prediction = model.predict([userTensor, movieTensor]);
        const predictionData = await prediction.data();
        let predictedRating = predictionData[0];
        
        // DOUBLE-CHECK BOUNDS (mathematically shouldn't be needed but ensures safety)
        predictedRating = Math.max(1.0, Math.min(5.0, predictedRating));
        
        // Cleanup
        tf.dispose([userTensor, movieTensor, prediction]);

        // Display results
        const movie = movies.find(m => m.id === movieId);
        const movieTitle = movie ? (movie.year ? `${movie.title} (${movie.year})` : movie.title) : `Movie ${movieId}`;
        
        let ratingClass = 'medium';
        if (predictedRating >= 4) ratingClass = 'high';
        else if (predictedRating <= 2) ratingClass = 'low';
        
        updateResult(
            `Predicted rating for User ${userId} on "${movieTitle}": <strong>${predictedRating.toFixed(2)}/5</strong>`,
            ratingClass
        );
        
    } catch (error) {
        console.error('Prediction error:', error);
        updateResult('Error making prediction: ' + error.message, 'low');
    }
}

// Test function to verify bounds and variance
async function testModelBounds() {
    if (!model) return;
    
    console.log('Testing model predictions...');
    
    // Test some predictions to verify they stay within bounds
    const testUsers = [1, 10, 50];
    const testMovies = [1, 100, 500];
    
    for (let user of testUsers) {
        const predictions = [];
        for (let movie of testMovies) {
            const userTensor = tf.tensor1d([user]);
            const movieTensor = tf.tensor1d([movie]);
            const prediction = model.predict([userTensor, movieTensor]);
            const rating = (await prediction.data())[0];
            predictions.push(rating);
            tf.dispose([userTensor, movieTensor, prediction]);
        }
        
        const min = Math.min(...predictions);
        const max = Math.max(...predictions);
        const range = max - min;
        
        console.log(`User ${user}: Min=${min.toFixed(2)}, Max=${max.toFixed(2)}, Range=${range.toFixed(2)}`);
        
        // Verify bounds
        if (min < 1.0 || max > 5.0) {
            console.error('BOUNDS VIOLATION DETECTED!');
        }
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

// Call this after training to verify everything works
async function verifyModel() {
    await testModelBounds();
}