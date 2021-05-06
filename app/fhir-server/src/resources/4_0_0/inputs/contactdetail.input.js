const {
	GraphQLString,
	GraphQLList,
	GraphQLInputObjectType,
} = require('graphql');

/**
 * @name exports
 * @summary ContactDetail Input Schema
 */
module.exports = new GraphQLInputObjectType({
	name: 'ContactDetail_Input',
	description:
		'Base StructureDefinition for ContactDetail Type: Specifies contact information for a person or organization.',
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
		_name: {
			type: require('./element.input.js'),
			description: 'The name of an individual to contact.',
		},
		name: {
			type: GraphQLString,
			description: 'The name of an individual to contact.',
		},
		telecom: {
			type: new GraphQLList(require('./contactpoint.input.js')),
			description:
				'The contact details for the individual (if a name was provided) or the organization.',
		},
	}),
});
