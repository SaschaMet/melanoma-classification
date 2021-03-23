const {
	GraphQLString,
	GraphQLList,
	GraphQLInputObjectType,
} = require('graphql');

/**
 * @name exports
 * @summary Encounterhospitalization Input Schema
 */
module.exports = new GraphQLInputObjectType({
	name: 'Encounterhospitalization_Input',
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
		preAdmissionIdentifier: {
			type: require('./identifier.input.js'),
			description: 'Pre-admission identifier.',
		},
		origin: {
			type: GraphQLString,
			description:
				'The location/organization from which the patient came before admission.',
		},
		admitSource: {
			type: require('./codeableconcept.input.js'),
			description:
				'From where patient was admitted (physician referral, transfer).',
		},
		reAdmission: {
			type: require('./codeableconcept.input.js'),
			description:
				'Whether this hospitalization is a readmission and why if known.',
		},
		dietPreference: {
			type: new GraphQLList(require('./codeableconcept.input.js')),
			description: 'Diet preferences reported by the patient.',
		},
		specialCourtesy: {
			type: new GraphQLList(require('./codeableconcept.input.js')),
			description: 'Special courtesies (VIP, board member).',
		},
		specialArrangement: {
			type: new GraphQLList(require('./codeableconcept.input.js')),
			description:
				'Any special requests that have been made for this hospitalization encounter, such as the provision of specific equipment or other things.',
		},
		destination: {
			type: GraphQLString,
			description: 'Location/organization to which the patient is discharged.',
		},
		dischargeDisposition: {
			type: require('./codeableconcept.input.js'),
			description: 'Category or kind of location after discharge.',
		},
	}),
});
