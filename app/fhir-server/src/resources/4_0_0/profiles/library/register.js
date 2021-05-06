const {
	LibraryCreateMutation,
	LibraryUpdateMutation,
	LibraryRemoveMutation,
} = require('./mutation');

const {
	LibraryQuery,
	LibraryListQuery,
	LibraryInstanceQuery,
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
		Library: LibraryQuery,
		LibraryList: LibraryListQuery,
	},
	/**
	 * Define Mutation Schema's here
	 * Each profile will need to define the supported mutations
	 * and these keys must be unique across the entire application, like routes
	 */
	mutation: {
		LibraryCreate: LibraryCreateMutation,
		LibraryUpdate: LibraryUpdateMutation,
		LibraryRemove: LibraryRemoveMutation,
	},
	/**
	 * These properties are so the core router can setup the approriate endpoint
	 * for a direct query against a resource
	 */
	instance: {
		name: 'Library',
		path: '/4_0_0/Library/:id',
		query: LibraryInstanceQuery,
	},
};
