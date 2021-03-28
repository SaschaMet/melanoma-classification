const {
	EncounterCreateMutation,
	EncounterUpdateMutation,
	EncounterRemoveMutation,
} = require('./mutation');

const {
	EncounterQuery,
	EncounterListQuery,
	EncounterInstanceQuery,
    EncounterByPatientIdQuery
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
		Encounter: EncounterQuery,
		EncounterList: EncounterListQuery,
        EncounterByPatientId: EncounterByPatientIdQuery,
	},
	/**
	 * Define Mutation Schema's here
	 * Each profile will need to define the supported mutations
	 * and these keys must be unique across the entire application, like routes
	 */
	mutation: {
		EncounterCreate: EncounterCreateMutation,
		EncounterUpdate: EncounterUpdateMutation,
		EncounterRemove: EncounterRemoveMutation,
	},
	/**
	 * These properties are so the core router can setup the approriate endpoint
	 * for a direct query against a resource
	 */
	instance: {
		name: 'Encounter',
		path: '/4_0_0/Encounter/:id',
		query: EncounterInstanceQuery,
	},
};
