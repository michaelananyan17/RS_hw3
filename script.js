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
        
        // Define model architecture
        const numLatentFactors = 8;
        
        const userEmbedding = tf.layers.embedding({
            inputDim: numUsers + 1,
            outputDim: numLatentFactors,
            inputLength: 1,
            name: 'user-embedding'
        });
        
        const movieEmbedding = tf.layers.embedding({
            inputDim: numMovies + 1,
            outputDim: numLatentFactors,
            inputLength: 1,
            name: 'movie-embedding'
        });

        const userInput = tf.layers.input({ shape: [1], name: 'user-input' });
        const movieInput = tf.layers.input({ shape: [1], name: 'movie-input' });

        const userVec = userEmbedding.apply(userInput);
        const movieVec = movieEmbedding.apply(movieInput);
        
        const userFlatten = tf.layers.flatten().apply(userVec);
        const movieFlatten = tf.layers.flatten().apply(movieVec);

        const dotProduct = tf.layers.dot({ axes: 1 }).apply([userFlatten, movieFlatten]);
        
        model = tf.model({ inputs: [userInput, movieInput], outputs: dotProduct });
        model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

        // Prepare data for training
        const userTensors = tf.tensor2d(ratings.map(r => r.userId), [ratings.length, 1]);
        const movieTensors = tf.tensor2d(ratings.map(r => r.movieId), [ratings.length, 1]);
        const ratingTensors = tf.tensor2d(ratings.map(r => r.rating), [ratings.length, 1]);

        // Train the model
        await model.fit([userTensors, movieTensors], ratingTensors, {
            epochs: 5,
            batchSize: 64,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    updateStatus(`Epoch ${epoch + 1}/5, Loss: ${logs.loss.toFixed(4)}`);
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
