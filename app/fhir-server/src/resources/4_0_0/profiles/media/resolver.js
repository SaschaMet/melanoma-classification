const errorUtils = require('../../../../utils/error.utils');
const {getCurrentDate} = require("../../../../utils/date")

/**
 * @name exports.getMedia
 * @static
 * @summary Media resolver.
 */
module.exports.getMedia = async function getMedia(root, args, ctx = {}, info) {
    let db = ctx.server.db;
    let version = ctx.version;
    let logger = ctx.server.logger;

    try {
        const response = await db.collection('media').findOne({ id: args._id })
        return response
    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(err);
        return errorUtils.formatErrorForGraphQL(error)
    }
};

/**
 * @name exports.getMediaByEncounterId
 * @static
 * @summary Encounter resolver.
 */
module.exports.getMediaByEncounterId = async function getMediaByEncounterId(
	root,
	args,
	ctx,
	info,
) {
    let db = ctx.server.db;
    let version = ctx.version;
    let logger = ctx.server.logger;
    try {
        const {_id} = args
        const query = { encounter: _id }
        const response = await db.collection('media').find(query).toArray()
        return {entries: response.map(r => r.id)}
    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(error);
    }
};

/**
 * @name exports.getMediaList
 * @static
 * @summary Media list resolver.
 */
module.exports.getMediaList = async function getMediaList(
	root,
	args,
	ctx = {},
	info,
) {
    let db = ctx.server.db;
    let version = ctx.version;
    let logger = ctx.server.logger;

    try {
        const response = await db.collection('media').find().toArray();
        return response
    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(err);
        return errorUtils.formatErrorForGraphQL(error)
    }
};

/**
 * @name exports.getMediaInstance
 * @static
 * @summary Media instance resolver.
 */
module.exports.getMediaInstance = async function getMediaInstance(
	root,
	args,
	context = {},
	info,
) {
	let { server, version, req, res } = context;
	return {};
};

/**
 * @name exports.createMedia
 * @static
 * @summary Create Media resolver.
 */
module.exports.createMedia = async function createMedia(
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
        const update = { $set: { id, ...resource, meta: { createdAt: getCurrentDate(), updatedAt: getCurrentDate() } }, };
        const options = { upsert: true, };
        await db.collection('media').updateOne(query, update, options);
        const response = await db.collection('media').findOne({ id })
        return response

    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(err);
        return errorUtils.formatErrorForGraphQL(error)
    }
};

/**
 * @name exports.updateMedia
 * @static
 * @summary Update Media resolver.
 */
module.exports.updateMedia = async function updateMedia(
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
        const update = { $set: { id, ...resource, meta: { createdAt: getCurrentDate(), updatedAt: getCurrentDate() } }, };
        const options = { upsert: true, };
        await db.collection('media').updateOne(query, update, options);
        const response = await db.collection('media').findOne({ id })
        return response
    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(err);
        return errorUtils.formatErrorForGraphQL(error)
    }
};

/**
 * @name exports.removeMedia
 * @static
 * @summary Remove Media resolver.
 */
module.exports.removeMedia = async function removeMedia(
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
        await db.collection('media').deleteOne({id});
        return { id }
    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(err);
        return errorUtils.formatErrorForGraphQL(error)
    }
};