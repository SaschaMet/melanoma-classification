const {
	GraphQLString,
	GraphQLList,
	GraphQLBoolean,
	GraphQLInputObjectType,
} = require('graphql');
const UriScalar = require('../scalars/uri.scalar.js');
const CodeScalar = require('../scalars/code.scalar.js');

/**
 * @name exports
 * @summary Coding Input Schema
 */
module.exports = new GraphQLInputObjectType({
	name: 'Coding_Input',
	description:
		'Base StructureDefinition for Coding Type: A reference to a code defined by a terminology system.',
	fields: () => ({
		_id: {
			type: require('./element.input.js'),
			description:
				'Unique id for the element within a resource (for internal references). This may be any string value that does not contain spaces.',
		},
		id: {
			type: GraphQLString,
			description:
				'Unique id for the element within a resource (for internal references). This may be any string value that does not contain spaces.',
		},
		extension: {
			type: new GraphQLList(require('./extension.input.js')),
			description:
				'May be used to represent additional information that is not part of the basic definition of the element. To make the use of extensions safe and manageable, there is a strict set of governance  applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension.',
		},
		_system: {
			type: require('./element.input.js'),
			description:
				'The identification of the code system that defines the meaning of the symbol in the code.',
		},
		system: {
			type: UriScalar,
			description:
				'The identification of the code system that defines the meaning of the symbol in the code.',
		},
		_version: {
			type: require('./element.input.js'),
			description:
				'The version of the code system which was used when choosing this code. Note that a well-maintained code system does not need the version reported, because the meaning of codes is consistent across versions. However this cannot consistently be assured, and when the meaning is not guaranteed to be consistent, the version SHOULD be exchanged.',
		},
		version: {
			type: GraphQLString,
			description:
				'The version of the code system which was used when choosing this code. Note that a well-maintained code system does not need the version reported, because the meaning of codes is consistent across versions. However this cannot consistently be assured, and when the meaning is not guaranteed to be consistent, the version SHOULD be exchanged.',
		},
		_code: {
			type: require('./element.input.js'),
			description:
				'A symbol in syntax defined by the system. The symbol may be a predefined code or an expression in a syntax defined by the coding system (e.g. post-coordination).',
		},
		code: {
			type: CodeScalar,
			description:
				'A symbol in syntax defined by the system. The symbol may be a predefined code or an expression in a syntax defined by the coding system (e.g. post-coordination).',
		},
		_display: {
			type: require('./element.input.js'),
			description:
				'A representation of the meaning of the code in the system, following the rules of the system.',
		},
		display: {
			type: GraphQLString,
			description:
				'A representation of the meaning of the code in the system, following the rules of the system.',
		},
		_userSelected: {
			type: require('./element.input.js'),
			description:
				'Indicates that this coding was chosen by a user directly - e.g. off a pick list of available items (codes or displays).',
		},
		userSelected: {
			type: GraphQLBoolean,
			description:
				'Indicates that this coding was chosen by a user directly - e.g. off a pick list of available items (codes or displays).',
		},
	}),
});
