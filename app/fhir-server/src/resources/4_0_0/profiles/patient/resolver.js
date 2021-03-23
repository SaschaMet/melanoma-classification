const errorUtils = require('../../../../utils/error.utils');
const {getCurrentDate} = require("../../../../utils/date")

module.exports.getPatient = async function getPatient(root, args, ctx, info) {
    let db = ctx.server.db;
    let version = ctx.version;
    let logger = ctx.server.logger;

    try {
        const response = await db.collection('patient').findOne({ id: args._id })
        return response
    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(err);
        return errorUtils.formatErrorForGraphQL(error)
    }
}

/**
 * @name exports.getPatientList
 * @static
 * @summary Patient list resolver.
 */
module.exports.getPatientList = async function getPatientList(
    root,
    args,
    ctx,
    info,
) {
    let db = ctx.server.db;
    let version = ctx.version;
    let logger = ctx.server.logger;

    try {
        const response = await db.collection('patient').find().toArray();
        return response
    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(err);
        return errorUtils.formatErrorForGraphQL(error)
    }
}

/**
 * @name exports.getAllPatients
 * @static
 * @summary Patient list resolver.
 */
module.exports.getAllPatients = async function getAllPatients(
    root,
    args,
    ctx,
    info,
) {
    let db = ctx.server.db;
    let version = ctx.version;
    let logger = ctx.server.logger;

    try {
        const response = await db.collection('patient').find().toArray();
        return {entries: response.map(r => r.id)}
    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(err);
        return errorUtils.formatErrorForGraphQL(error)
    }
}

/**
 * @name exports.getPatientInstance
 * @static
 * @summary Patient instance resolver.
 */
module.exports.getPatientInstance = function getPatientInstance(
    root,
    args,
    ctx,
    info,
) {
    let { server, version, req, res } = context;
    return {};
};

/**
 * @name exports.createPatient
 * @static
 * @summary Create Patient resolver.
 */
module.exports.createPatient = async function createPatient(
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
        await db.collection('patient').updateOne(query, update, options);
        const response = await db.collection('patient').findOne({ id })
        return response

    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(err);
        return errorUtils.formatErrorForGraphQL(error)
    }
};

/**
 * @name exports.updatePatient
 * @static
 * @summary Update Patient resolver.
 */
module.exports.updatePatient = async function updatePatient(
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
        const update = { $set: { id, ...resource, meta: { updatedAt: getCurrentDate() } }, };
        const options = { upsert: true, };
        await db.collection('patient').updateOne(query, update, options);
        const response = await db.collection('patient').findOne({ id })
        return response
    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(err);
        return errorUtils.formatErrorForGraphQL(error)
    }
};

/**
 * @name exports.removePatient
 * @static
 * @summary Remove Patient resolver.
 */
module.exports.removePatient = async function removePatient(
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
        await db.collection('patient').deleteOne({id});
        return { id }
    } catch (err) {
        let error = errorUtils.internal(version, err.message);
        logger.error(err);
        return errorUtils.formatErrorForGraphQL(error)
    }
};
