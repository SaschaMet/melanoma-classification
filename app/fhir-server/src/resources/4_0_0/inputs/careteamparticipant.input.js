const {
	GraphQLString,
	GraphQLList,
	GraphQLInputObjectType,
} = require('graphql');

/**
 * @name exports
 * @summary CareTeamparticipant Input Schema
 */
module.exports = new GraphQLInputObjectType({
	name: 'CareTeamparticipant_Input',
	description: '',
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
		modifierExtension: {
			type: new GraphQLList(require('./extension.input.js')),
			description:
				"May be used to represent additional information that is not part of the basic definition of the element and that modifies the understanding of the element in which it is contained and/or the understanding of the containing element's descendants. Usually modifier elements provide negation or qualification. To make the use of extensions safe and manageable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension. Applications processing a resource are required to check for modifier extensions.  Modifier extensions SHALL NOT change the meaning of any elements on Resource or DomainResource (including cannot change the meaning of modifierExtension itself).",
		},
		role: {
			type: new GraphQLList(require('./codeableconcept.input.js')),
			description:
				"Indicates specific responsibility of an individual within the care team, such as 'Primary care physician', 'Trained social worker counselor', 'Caregiver', etc.",
		},
		member: {
			type: GraphQLString,
			description:
				'The specific person or organization who is participating/expected to participate in the care team.',
		},
		onBehalfOf: {
			type: GraphQLString,
			description: 'The organization of the practitioner.',
		},
		period: {
			type: require('./period.input.js'),
			description:
				'Indicates when the specific member or organization did (or is intended to) come into effect and end.',
		},
	}),
});
