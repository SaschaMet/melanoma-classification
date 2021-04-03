const { GraphQLString } = require('graphql');
const TokenScalar = require('../scalars/token.scalar.js');
const DateScalar = require('../scalars/date.scalar.js');

/**
 * @name exports
 * @static
 * @summary Arguments for the bundle query
 */
module.exports = {
	// http://hl7.org/fhir/SearchParameter/Bundle-composition
	composition: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'Bundle.entry[0].resource',
		description:
			"The first resource in the bundle, if the bundle type is 'document' - this is a composition, and this parameter provides access to search its contents",
	},
	// http://hl7.org/fhir/SearchParameter/Bundle-identifier
	identifier: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'Bundle.identifier',
		description: 'Persistent identifier for the bundle',
	},
	// http://hl7.org/fhir/SearchParameter/Bundle-message
	message: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'Bundle.entry[0].resource',
		description:
			"The first resource in the bundle, if the bundle type is 'message' - this is a message header, and this parameter provides access to search its contents",
	},
	// http://hl7.org/fhir/SearchParameter/Bundle-timestamp
	timestamp: {
		type: DateScalar,
		fhirtype: 'date',
		xpath: 'Bundle.timestamp',
		description: 'When the bundle was assembled',
	},
	// http://hl7.org/fhir/SearchParameter/Bundle-type
	type: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'Bundle.type',
		description:
			'document | message | transaction | transaction-response | batch | batch-response | history | searchset | collection',
	},
};
