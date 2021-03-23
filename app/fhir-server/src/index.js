const { SERVER_CONFIG } = require('./config.js');
const { DB_NAME, DBURL } = require('../secrets.js');
const FHIRServer = require('./server.js');

// load environment settings
require('./environment.js');

// Start buliding our server
const server = new FHIRServer(SERVER_CONFIG)
	.configureMiddleware()
	// .configurePassport()
	.configureHelmet()
	.enableHealthCheck()
	.setProfileRoutes()
	.setErrorRoutes();

server.initializeDatabaseConnection({
    url: DBURL,
    db_name: DB_NAME,
    mongo_options: { auto_reconnect: true, useNewUrlParser: true, useUnifiedTopology: true }
}).then(() => {
    server.listen(SERVER_CONFIG.port);
    server.logger.info('FHIR Server listening on localhost:' + SERVER_CONFIG.port);
}).catch(err => {
    console.error(err);
    server.logger.error('Fatal Error connecting to Mongo.');
});
