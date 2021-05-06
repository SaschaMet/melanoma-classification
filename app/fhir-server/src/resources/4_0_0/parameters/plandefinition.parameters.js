const { GraphQLString } = require('graphql');
const TokenScalar = require('../scalars/token.scalar.js');
const DateScalar = require('../scalars/date.scalar.js');
const UriScalar = require('../scalars/uri.scalar.js');

/**
 * @name exports
 * @static
 * @summary Arguments for the plandefinition query
 */
module.exports = {
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-composed-of
	composed_of: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: "PlanDefinition.relatedArtifact[type/@value='composed-of'].resource",
		description: 'What resource is being referenced',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-context
	context: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'PlanDefinition.useContext.valueCodeableConcept',
		description: 'A use context assigned to the plan definition',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-context-quantity
	context_quantity: {
		type: GraphQLString,
		fhirtype: 'quantity',
		xpath: 'PlanDefinition.useContext.valueQuantity',
		description:
			'A quantity- or range-valued use context assigned to the plan definition',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-context-type
	context_type: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'PlanDefinition.useContext.code',
		description: 'A type of use context assigned to the plan definition',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-date
	date: {
		type: DateScalar,
		fhirtype: 'date',
		xpath: 'PlanDefinition.date',
		description: 'The plan definition publication date',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-definition
	definition: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'PlanDefinition.action.definitionCanonical',
		description: 'Activity or plan definitions used by plan definition',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-depends-on
	depends_on: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: "PlanDefinition.relatedArtifact[type/@value='depends-on'].resource",
		description: 'What resource is being referenced',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-derived-from
	derived_from: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath:
			"PlanDefinition.relatedArtifact[type/@value='derived-from'].resource",
		description: 'What resource is being referenced',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-description
	description: {
		type: GraphQLString,
		fhirtype: 'string',
		xpath: 'PlanDefinition.description',
		description: 'The description of the plan definition',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-effective
	effective: {
		type: DateScalar,
		fhirtype: 'date',
		xpath: 'PlanDefinition.effectivePeriod',
		description:
			'The time during which the plan definition is intended to be in use',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-identifier
	identifier: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'PlanDefinition.identifier',
		description: 'External identifier for the plan definition',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-jurisdiction
	jurisdiction: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'PlanDefinition.jurisdiction',
		description: 'Intended jurisdiction for the plan definition',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-name
	name: {
		type: GraphQLString,
		fhirtype: 'string',
		xpath: 'PlanDefinition.name',
		description: 'Computationally friendly name of the plan definition',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-predecessor
	predecessor: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: "PlanDefinition.relatedArtifact[type/@value='predecessor'].resource",
		description: 'What resource is being referenced',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-publisher
	publisher: {
		type: GraphQLString,
		fhirtype: 'string',
		xpath: 'PlanDefinition.publisher',
		description: 'Name of the publisher of the plan definition',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-status
	status: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'PlanDefinition.status',
		description: 'The current status of the plan definition',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-successor
	successor: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: "PlanDefinition.relatedArtifact[type/@value='successor'].resource",
		description: 'What resource is being referenced',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-title
	title: {
		type: GraphQLString,
		fhirtype: 'string',
		xpath: 'PlanDefinition.title',
		description: 'The human-friendly name of the plan definition',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-topic
	topic: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'PlanDefinition.topic',
		description: 'Topics associated with the module',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-type
	type: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'PlanDefinition.type',
		description:
			'The type of artifact the plan (e.g. order-set, eca-rule, protocol)',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-url
	url: {
		type: UriScalar,
		fhirtype: 'uri',
		xpath: 'PlanDefinition.url',
		description: 'The uri that identifies the plan definition',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-version
	version: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'PlanDefinition.version',
		description: 'The business version of the plan definition',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-context-type-quantity
	context_type_quantity: {
		type: GraphQLString,
		fhirtype: 'composite',
		xpath: '',
		description:
			'A use context type and quantity- or range-based value assigned to the plan definition',
	},
	// http://hl7.org/fhir/SearchParameter/PlanDefinition-context-type-value
	context_type_value: {
		type: GraphQLString,
		fhirtype: 'composite',
		xpath: '',
		description: 'A use context type and value assigned to the plan definition',
	},
};
