const {getCurrentDate} = require("../../../../utils/date")
const errorUtils = require('../../../../utils/error.utils');

/**
 * @name exports.getEncounter
 * @static
 * @summary Encounter resolver.
 */
module.exports.getEncounter = async function getEncounter(
	root,
	args,
	ctx,
	info,
) {
    let db = ctx.server.db;
    let version = ctx.version;
    let logger = ctx.server.logger;

    try {
        const response = await db.collection('encounter').findOne({ id: args._id })
        return response
    } catch (err) {
        logger.error(err);
        let error = errorUtils.internal(version, err.message);
    }
};

/**
 * @name exports.getEncounterByPatientId
 * @static
 * @summary Encounter resolver.
 */
module.exports.getEncounterByPatientId = async function getEncounterByPatientId(
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
        const query = { subject: _id }
        const response = await db.collection('encounter').find(query).toArray()
        return {entries: response.map(r => r.id)}
    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(error);
    }
};

/**
 * @name exports.getEncounterList
 * @static
 * @summary Encounter list resolver.
 */
module.exports.getEncounterList = async function getEncounterList(
	root,
	args,
	ctx = {},
	info,
) {
    let db = ctx.server.db;
    let version = ctx.version;
    let logger = ctx.server.logger;

    try {
        const response = await db.collection('encounter').find().toArray();
        return response
    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(err);
        return errorUtils.formatErrorForGraphQL(error)
    }
};

/**
 * @name exports.getEncounterInstance
 * @static
 * @summary Encounter instance resolver.
 */
module.exports.getEncounterInstance = async function getEncounterInstance(
	root,
	args,
	context = {},
	info,
) {
	let { server, version, req, res } = context;
	return {};
};

/**
 * @name exports.createEncounter
 * @static
 * @summary Create Encounter resolver.
 */
module.exports.createEncounter = async function createEncounter(
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
        await db.collection('encounter').updateOne(query, update, options);
        const response = await db.collection('encounter').findOne({ id })
        return response

    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(err);
        return errorUtils.formatErrorForGraphQL(error)
    }
};

/**
 * @name exports.updateEncounter
 * @static
 * @summary Update Encounter resolver.
 */
module.exports.updateEncounter = async function updateEncounter(
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
        const update = { $set: { id, ...resource, meta: { updatedAt: getCurrentDate() }  }, };
        const options = { upsert: true, };
        await db.collection('encounter').updateOne(query, update, options);
        const response = await db.collection('encounter').findOne({ id })
        return response
    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(err);
        return errorUtils.formatErrorForGraphQL(error)
    }
};

/**
 * @name exports.removeEncounter
 * @static
 * @summary Remove Encounter resolver.
 */
module.exports.removeEncounter = async function removeEncounter(
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
        await db.collection('encounter').deleteOne({id});
        return { id }
    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(err);
        return errorUtils.formatErrorForGraphQL(error)
    }
};