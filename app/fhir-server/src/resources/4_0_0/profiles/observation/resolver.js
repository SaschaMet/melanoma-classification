/**
 * @name exports.getObservation
 * @static
 * @summary Observation resolver.
 */
module.exports.getObservation = async function getObservation(
	root,
	args,
	ctx,
	info,
) {
    let db = ctx.server.db;
    let version = ctx.version;
    let logger = ctx.server.logger;

    try {
        const response = await db.collection('observation').findOne({ id: args._id })
        return response
    } catch (err) {
        logger.error(err);
        let error = errorUtils.internal(version, err.message);
        return errorUtils.formatErrorForGraphQL(error)
    }
};

/**
 * @name exports.getObservationList
 * @static
 * @summary Observation list resolver.
 */
module.exports.getObservationList = async function getObservationList(
	root,
	args,
	ctx = {},
	info,
) {
    let db = ctx.server.db;
    let version = ctx.version;
    let logger = ctx.server.logger;

    try {
        const response = await db.collection('observation').find().toArray();
        return response
    } catch (err) {
        logger.error(err);
        let error = errorUtils.internal(version, err.message);
        return errorUtils.formatErrorForGraphQL(error)
    }
};

/**
 * @name exports.getObservationInstance
 * @static
 * @summary Observation instance resolver.
 */
module.exports.getObservationInstance = async function getObservationInstance(
	root,
	args,
	context = {},
	info,
) {
	let { server, version, req, res } = context;
	return {};
};

/**
 * @name exports.createObservation
 * @static
 * @summary Create Observation resolver.
 */
module.exports.createObservation = async function createObservation(
    root,
    args,
    ctx,
    info,
) {
    let db = ctx.server.db;
    let version = ctx.version;
    let logger = ctx.server.logger;

    try {
        const {id, resource } = args
        delete resource.resourceType

        const query = { id, };
        const update = { $set: { id, ...resource, }, };
        const options = { upsert: true, };
        await db.collection('observation').updateOne(query, update, options);
        const response = await db.collection('observation').findOne({ id })
        return response

    } catch (err) {
        logger.error(err);
        let error = errorUtils.internal(version, err.message);
        return errorUtils.formatErrorForGraphQL(error)
    }
};

/**
 * @name exports.updateObservation
 * @static
 * @summary Update Observation resolver.
 */
module.exports.updateObservation = async function updateObservation(
    root,
    args,
    ctx,
    info,
) {

    let db = ctx.server.db;
    let version = ctx.version;
    let logger = ctx.server.logger;

    try {
        const {id, resource } = args
        delete resource.resourceType

        const query = { id, };
        const update = { $set: { id, ...resource, }, };
        const options = { upsert: true, };
        await db.collection('observation').updateOne(query, update, options);
        const response = await db.collection('observation').findOne({ id })
        return response
    } catch (err) {
        logger.error(err);
        let error = errorUtils.internal(version, err.message);
        return errorUtils.formatErrorForGraphQL(error)
    }
};

/**
 * @name exports.removeObservation
 * @static
 * @summary Remove Observation resolver.
 */
module.exports.removeObservation = async function removeObservation(
    root,
    args,
    ctx,
    info,
) {

    let db = ctx.server.db;
    let version = ctx.version;
    let logger = ctx.server.logger;

    try {
        const { id } = args
        await db.collection('observation').deleteOne({id});
        return { id }
    } catch (err) {
        logger.error(err);
        let error = errorUtils.internal(version, err.message);
        return errorUtils.formatErrorForGraphQL(error)
    }
};
