const {
	GraphQLString,
	GraphQLList,
	GraphQLFloat,
	GraphQLInt,
	GraphQLObjectType,
} = require('graphql');

/**
 * @name exports
 * @summary ObservationDefinitionquantitativeDetails Schema
 */
module.exports = new GraphQLObjectType({
	name: 'ObservationDefinitionquantitativeDetails',
	description: '',
	fields: () => ({
		_id: {
			type: require('./element.schema.js'),
			description:
				'Unique id for the element within a resource (for internal references). This may be any string value that does not contain spaces.',
		},
		id: {
			type: GraphQLString,
			description:
				'Unique id for the element within a resource (for internal references). This may be any string value that does not contain spaces.',
		},
		extension: {
			type: new GraphQLList(require('./extension.schema.js')),
			description:
				'May be used to represent additional information that is not part of the basic definition of the element. To make the use of extensions safe and manageable, there is a strict set of governance  applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension.',
		},
		modifierExtension: {
			type: new GraphQLList(require('./extension.schema.js')),
			description:
				"May be used to represent additional information that is not part of the basic definition of the element and that modifies the understanding of the element in which it is contained and/or the understanding of the containing element's descendants. Usually modifier elements provide negation or qualification. To make the use of extensions safe and manageable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension. Applications processing a resource are required to check for modifier extensions.  Modifier extensions SHALL NOT change the meaning of any elements on Resource or DomainResource (including cannot change the meaning of modifierExtension itself).",
		},
		customaryUnit: {
			type: require('./codeableconcept.schema.js'),
			description:
				'Customary unit used to report quantitative results of observations conforming to this ObservationDefinition.',
		},
		unit: {
			type: require('./codeableconcept.schema.js'),
			description:
				'SI unit used to report quantitative results of observations conforming to this ObservationDefinition.',
		},
		_conversionFactor: {
			type: require('./element.schema.js'),
			description:
				'Factor for converting value expressed with SI unit to value expressed with customary unit.',
		},
		conversionFactor: {
			type: GraphQLFloat,
			description:
				'Factor for converting value expressed with SI unit to value expressed with customary unit.',
		},
		_decimalPrecision: {
			type: require('./element.schema.js'),
			description:
				'Number of digits after decimal separator when the results of such observations are of type Quantity.',
		},
		decimalPrecision: {
			type: GraphQLInt,
			description:
				'Number of digits after decimal separator when the results of such observations are of type Quantity.',
		},
	}),
});
