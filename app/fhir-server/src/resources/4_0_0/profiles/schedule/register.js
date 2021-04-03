const {
	ScheduleCreateMutation,
	ScheduleUpdateMutation,
	ScheduleRemoveMutation,
} = require('./mutation');

const {
	ScheduleQuery,
	ScheduleListQuery,
	ScheduleInstanceQuery,
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
		Schedule: ScheduleQuery,
		ScheduleList: ScheduleListQuery,
	},
	/**
	 * Define Mutation Schema's here
	 * Each profile will need to define the supported mutations
	 * and these keys must be unique across the entire application, like routes
	 */
	mutation: {
		ScheduleCreate: ScheduleCreateMutation,
		ScheduleUpdate: ScheduleUpdateMutation,
		ScheduleRemove: ScheduleRemoveMutation,
	},
	/**
	 * These properties are so the core router can setup the approriate endpoint
	 * for a direct query against a resource
	 */
	instance: {
		name: 'Schedule',
		path: '/4_0_0/Schedule/:id',
		query: ScheduleInstanceQuery,
	},
};
