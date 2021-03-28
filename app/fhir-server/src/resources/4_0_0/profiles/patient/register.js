const {
	PatientCreateMutation,
	PatientUpdateMutation,
	PatientRemoveMutation,
} = require('./mutation');

const {
	PatientQuery,
    AllPatientsQuery,
	PatientListQuery,
	PatientInstanceQuery,
} = require('./query');

/**
 * @name exports
 * @static
 * @summary GraphQL Configurations. This is needed to register this profile
 * with the GraphQL server.
 */
module.exports = {
	/**
	 * Define Query Schema's here
	 * Each profile will need to define the two queries it supports
	 * and these keys must be unique across the entire application, like routes
	 */
	query: {
		Patient: PatientQuery,
		PatientList: PatientListQuery,
        AllPatientsQuery: AllPatientsQuery,
	},
	/**
	 * Define Mutation Schema's here
	 * Each profile will need to define the supported mutations
	 * and these keys must be unique across the entire application, like routes
	 */
	mutation: {
		PatientCreate: PatientCreateMutation,
		PatientUpdate: PatientUpdateMutation,
		PatientRemove: PatientRemoveMutation,
	},
	/**
	 * These properties are so the core router can setup the approriate endpoint
	 * for a direct query against a resource
	 */
	instance: {
		name: 'Patient',
		path: '/4_0_0/Patient/:id',
		query: PatientInstanceQuery,
	},
};
